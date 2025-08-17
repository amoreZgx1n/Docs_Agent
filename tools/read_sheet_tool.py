import asyncio
import google.auth
import json
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

def read_google_sheet(spreadsheet_id: str, range_name: str) -> str:
    """
    用于从指定的 Google Sheet 中读取数据。
    :param spreadsheet_id: Google Sheet 的 ID.
    :param range_name: 要读取的范围，例如 'Sheet1!A1:B5'.
    :return: 以list形式返回读取到的表格数据，如果出错则返回错误信息。
    """
    print(f"--- 正在执行工具: read_google_sheet ---")
    print(f"--- 参数: spreadsheet_id='{spreadsheet_id}', range_name='{range_name}' ---")
    
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    
    try:
        creds, _ = google.auth.default(scopes=SCOPES)
        service = build("sheets", "v4", credentials=creds)

        sheet = service.spreadsheets()
        result = (
            sheet.values()
            .get(spreadsheetId=spreadsheet_id, range=range_name)
            .execute()
        )
        values = result.get("values", [])
        if not values:
            return "表格中没有找到数据。"
        values_json = convert_list_to_json(values)
        print(f"转换后的JSON数据: {values_json}")

        return values_json

    except HttpError as err:
        return f"访问 Google Sheet 时发生错误: {err}"
    except Exception as e:
        return f"发生未知错误: {e}"

def convert_list_to_json(data: list) -> str:
    """
    将包含表头的二维列表转换为JSON对象数组格式的字符串。

    :param data: 输入的二维列表，第一行必须是表头（keys）。
    :return: 格式化后的JSON字符串。如果输入数据不合法，则返回一个空数组的JSON字符串。
    """
    if not data or len(data) < 2:
        return "[]"

    headers = data[0]
    rows = data[1:]
    json_data = [dict(zip(headers, row)) for row in rows]
    return json.dumps(json_data, ensure_ascii=False, indent=4)

async def main():
    # --- 将函数封装成 LlamaIndex 工具 ---
    sheet_tool = FunctionTool.from_defaults(
        fn=read_google_sheet,
        name="google_sheet_reader", # 给工具起一个明确的名字
        description="这个工具可以从指定的Google Sheet ID和范围中读取数据并返回内容。"
    )

    # --- 创建并配置 Agent ---
    agent = FunctionAgent(
        tools=[sheet_tool],
        llm=Ollama(model="llama3.1", request_timeout=360.0),
        system_prompt="你是一个数据分析助手。你可以使用工具读取Google Sheet中的数据来回答问题。",
        verbose=True # 强烈建议在调试时开启，以观察Agent的思考过程
    )

    # --- 向 Agent 提问 ---
    spreadsheet_id = "1ZPNqJhGBHDtBQ8D9GMgvva3uj0_v1KyFaU_7ww3RhAA"
    range_name = "grade!A:Z"    # 读取全部字段
    prompt = f"请使用工具读取ID为 '{spreadsheet_id}' 的电子表格中名为 '{range_name}' 的工作表，然后告诉我学号为20250102的学生姓名，以及他的班级排名是多少？"
    
    response = await agent.run(prompt)
    print("✅ Agent 的最终回答:")
    print(str(response))

if __name__ == "__main__":
    asyncio.run(main())