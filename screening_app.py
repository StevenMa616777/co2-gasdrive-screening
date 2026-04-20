import io

import pandas as pd
import streamlit as st

from screening_service import (
    MODEL_METRICS_PATH,
    RAW_INPUT_COLUMNS,
    build_template_dataframe,
    dataframe_to_excel_bytes,
    load_bundle,
    score_dataframe,
)


st.set_page_config(page_title="气驱潜力初筛工具", page_icon="⛽", layout="wide")


@st.cache_data(show_spinner=False)
def load_template_bytes() -> bytes:
    return dataframe_to_excel_bytes(build_template_dataframe())


@st.cache_data(show_spinner=False)
def score_uploaded_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return score_dataframe(df)


bundle = load_bundle()
metrics = bundle["metrics"]

st.title("气驱潜力初筛工具")
st.caption("上传井数据后，系统会自动完成油藏参数清洗、特征重组、潜力评分和关键原因解释。")

with st.sidebar:
    st.subheader("模型范围")
    st.write("这版工具主模型使用层位、厚度/射孔关系、储层物性和地层状态参数。")
    st.write("密度和粘度只作为辅助信息，不再作为主 backbone 的必选输入。")
    st.write(f"当前训练样本: {metrics['data_summary']['trainable_rows_after_quality_filter']}")
    st.write(f"分类 Macro-F1: {metrics['classifier']['holdout_metrics']['f1_macro']:.3f}")
    st.write(f"分类 Balanced-Acc: {metrics['classifier']['holdout_metrics']['balanced_accuracy']:.3f}")
    st.download_button(
        "下载输入模板",
        data=load_template_bytes(),
        file_name="气驱潜力初筛模板.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.code("streamlit run screening_app.py", language="bash")


uploaded_file = st.file_uploader("上传待筛选数据", type=["xlsx", "xls", "csv"])

if uploaded_file is None:
    st.info("模板列至少需要包含这些字段。")
    st.dataframe(pd.DataFrame({"输入字段": RAW_INPUT_COLUMNS}), use_container_width=True, hide_index=True)
else:
    if uploaded_file.name.lower().endswith(".csv"):
        source_df = pd.read_csv(uploaded_file)
    else:
        workbook = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("选择工作表", workbook.sheet_names, index=0)
        source_df = workbook.parse(sheet_name)

    st.subheader("原始数据预览")
    st.dataframe(source_df.head(20), use_container_width=True)

    if st.button("开始筛选", type="primary"):
        with st.spinner("正在清洗数据并计算潜力评分..."):
            scored_df = score_uploaded_dataframe(source_df)

        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        summary_col1.metric("样本数", len(scored_df))
        summary_col2.metric("高价值响应井", int((scored_df["潜力分级"] == "高价值响应").sum()))
        summary_col3.metric("低增效井", int((scored_df["潜力分级"] == "低增效").sum()))
        summary_col4.metric("低置信度样本", int((scored_df["screening_confidence"] < 0.5).sum()))

        st.subheader("结果预览")
        preview_columns = [
            column
            for column in ["井号", "潜力分级", "筛选建议", "响应概率", "高价值响应概率", "潜力评分", "screening_confidence", "关键原因1", "关键原因2", "关键原因3", "数据质量提示"]
            if column in scored_df.columns
        ]
        st.dataframe(
            scored_df[preview_columns].sort_values(["潜力分级", "潜力评分"], ascending=[True, False]),
            use_container_width=True,
        )

        output_bytes = dataframe_to_excel_bytes(scored_df)
        st.download_button(
            "下载完整筛选结果",
            data=output_bytes,
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_screened.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.subheader("字段说明")
        st.markdown(
            "\n".join(
                [
                    "- `潜力评分`: 由三个分桶类别概率加权得到的综合排序分数。",
                    "- `潜力分级`: 当前分成 `无效响应 / 低增效 / 高价值响应` 三个桶。",
                    "- `高价值响应概率`: 模型判断该井落入高价值响应桶的概率。",
                    "- `screening_confidence`: 根据参数完整度和异常值情况生成的置信度。",
                    "- `关键原因1~3`: 系统基于历史响应规律给出的主要判断依据。",
                    "- `数据质量提示`: 告诉你哪些井更适合人工复核。",
                ]
            )
        )
