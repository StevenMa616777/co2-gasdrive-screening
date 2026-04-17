# 气驱潜力初筛系统

一个基于油藏条件的气驱潜力初筛工具，支持数据清洗、特征工程、三桶分类预测，以及网页和命令行两种使用方式。

## 功能概览

- 面向油藏状况进行初筛，不依赖注入施工参数或现场增油结果等泄漏特征
- 自动完成区间解析、异常值识别、缺失信息降权和特征重组
- 输出 `潜力分级`、`筛选建议`、`响应概率`、`高价值响应概率`、`screening_confidence`、`关键原因1~3`、`数据质量提示`
- 当前默认策略采用平衡型阈值，适合在尚未接入经济效益模型前作为基础筛选规则

## 主要文件

- `screening_pipeline.py`：训练、评估和产物导出
- `screening_service.py`：模型加载、打分和解释逻辑
- `screening_tool.py`：命令行工具
- `screening_app.py`：Streamlit 网页工具
- `气驱潜力初筛模板.xlsx`：空白导入模板
- `测试案例_10条_多场景.xlsx`：示例测试数据

## 本地网页

```bash
.venv/bin/streamlit run screening_app.py
```

默认打开 [http://localhost:8501](http://localhost:8501)。

## 命令行

```bash
.venv/bin/python screening_tool.py 数据.xlsx --output 结果.xlsx
```

导出模板：

```bash
.venv/bin/python screening_tool.py --template 气驱潜力初筛模板.xlsx
```

## 说明

- 仓库默认不包含训练输出、模型产物和原始业务数据
- 如需重新训练，请在本地准备原始 Excel 后运行 `screening_pipeline.py`
