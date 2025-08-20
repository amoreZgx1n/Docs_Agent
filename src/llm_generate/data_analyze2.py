from src.llm_generate.code_generate import Code_Generator
from src.llm_generate.llm import LLM
import pandas as pd
from src.llm_generate.df_summary import DF_Summary
from src.llm_generate.code_executor import execute

def analyze_data(user_query: str, df: pd.DataFrame):
    """
    分析用户查询并生成数据分析报告
    
    该函数通过以下步骤完成数据分析：
    1. 初始化大语言模型
    2. 生成数据框摘要信息
    3. 根据用户查询和数据摘要生成分析代码
    4. 执行生成的代码获得分析结果
    5. 基于分析结果生成最终的分析报告
    
    参数:
        user_query (str): 用户的数据分析查询问题
        df (pd.DataFrame): 需要分析的数据框
        
    返回:
        str: 以markdown格式返回的数据分析报告
    """
    # 初始化大语言模型
    llm = LLM(backend="ollama")
    
    # 获取数据框的摘要信息
    df_summary = DF_Summary(df).get_summary()
    
    # 根据用户查询和数据摘要生成分析代码
    code = Code_Generator(llm).generate_code(user_query, df_summary)
    
    # 执行生成的代码获得分析结果数据框
    result_df = execute(df, code)
    
    # 构造分析报告生成提示词
    prompt = f"""
    你是一个数据分析专家。请你根据用户问题，结合数据信息，进行充分分析，给出分析报告。
    #用户问题
    {user_query}
    #数据信息   
    {result_df}

    要求以markdown形式返回，报告里要嵌有数据信息。
    """

    # 生成最终的数据分析报告
    analyze_result = llm.complete(prompt)
    return analyze_result

