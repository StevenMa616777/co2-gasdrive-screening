import argparse
import os
from datetime import datetime

from screening_service import build_template_dataframe, dataframe_to_excel_bytes, score_file


def default_output_path(input_path: str) -> str:
    stem, _ = os.path.splitext(os.path.basename(input_path))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(os.getcwd(), f"{stem}_screened_{timestamp}.xlsx")


def main() -> None:
    parser = argparse.ArgumentParser(description="油藏气驱潜力初筛工具")
    parser.add_argument("input", nargs="?", help="待筛选的 Excel/CSV 文件路径")
    parser.add_argument("--sheet", default=0, help="Excel 工作表名或序号，默认 0")
    parser.add_argument("--output", help="输出结果路径，默认自动生成 xlsx")
    parser.add_argument("--template", help="导出输入模板到指定路径后退出")
    args = parser.parse_args()

    if args.template:
        template_df = build_template_dataframe()
        with open(args.template, "wb") as file:
            file.write(dataframe_to_excel_bytes(template_df))
        print(f"模板已生成: {args.template}")
        return

    if not args.input:
        raise SystemExit("请提供输入文件路径，或使用 --template 导出模板。")

    output_path = args.output or default_output_path(args.input)
    score_file(args.input, output_path, sheet_name=args.sheet)
    print(f"筛选结果已写出: {output_path}")


if __name__ == "__main__":
    main()
