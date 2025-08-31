import json

def convert_list_to_json(data: list) -> str:
    """
    将包含表头的二维列表转换为JSON对象数组格式的字符串。

    :param data: 输入的二维列表，第一行必须是表头（keys）。
    :return: 格式化后的JSON字符串。如果输入数据不合法，则返回一个空数组的JSON字符串。
    """
    if not data or len(data) < 2:
        # 如果没有数据或只有表头，返回空列表
        return "[]"

    # 第一行是表头（keys）
    headers = data[0]
    # 后面的都是数据行（values）
    rows = data[1:]

    # 使用列表推导式高效地构建对象列表
    json_data = [dict(zip(headers, row)) for row in rows]

    # 使用 json.dumps 将Python列表转换为JSON格式的字符串
    # ensure_ascii=False 确保中文字符能正常显示
    # indent=4 让输出的JSON格式更美观，易于阅读
    return json.dumps(json_data, ensure_ascii=False, indent=4)

# --- 使用您提供的数据进行测试 ---
if __name__ == "__main__":
    # 您提供的表格数据
    sheet_data = [
        ['学号', '姓名', '班级', '性别', '考试名称', '考试日期', '语文', '数学', '英语', '物理', '化学', '生物', '历史', '总分', '平均分', '班级排名'],
        ['20250101', '赵博', '高一(2)班', '男', '2025学年第一学期期中考试', '2025-04-15', '124', '118', '128', '88', '86', '83', '89', '716', '102.3', '17'],
        ['20250102', '钱娟', '高一(1)班', '女', '2025学年第一学期期中考试', '2025-04-15', '133', '121', '135', '92', '90', '88', '94', '753', '107.6', '4'],
        # ... 此处省略了您提供的其余数据行
    ]

    # 调用转换函数
    json_output = convert_list_to_json(sheet_data)

    # 打印结果
    print(json_output)
