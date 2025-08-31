from src.llm_generate.llm import LLM
from datetime import datetime
import os
import re 
import json

class Code_Generator:
    def __init__(self, llm: LLM) -> None:
        """
        初始化代码生成器实例
        
        Args:
            llm (LLM): 大语言模型实例，用于代码生成
        """
        self.llm = llm
    
    def _load_prompt_tmpl(self):
        """
        加载代码生成的提示模板文件
        
        Returns:
            str: 从文件中读取的提示模板内容
        """
        version = "code_generate_prompt"
        with open(os.path.join('src','llm_generate','prompt', f"{version}.md"), encoding='utf-8') as f:
            prompt_tmpl = f.read()
        return prompt_tmpl
    
    def get_prompt(self, question: str, data_summary: dict):
        """
        根据问题和数据摘要生成完整的提示词
        
        Args:
            question (str): 用户提出的问题
            data_summary (dict): 数据摘要信息，包含head、info和describe字段
            
        Returns:
            str: 完整的提示词内容
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        prompt = self._load_prompt_tmpl().replace(
            '{{question}}', question
        ).replace(
            '{{current_time}}', current_time
        ).replace(
            '{{data_info}}', str(data_summary['head']) + '\n' + str(data_summary['info']) + '\n' + str(data_summary['describe'])
        )
        return prompt
    
    def generate_code(self, question: str, data_summary: dict):
        """
        根据问题和数据摘要生成代码
        
        Args:
            question (str): 用户提出的问题
            data_summary (dict): 数据摘要信息
            
        Returns:
            str: 从LLM响应中提取的代码
        """
        prompt = self.get_prompt(question, data_summary)
        resp = self.llm.complete(prompt)
        code = self.extract_code(resp, lang='python')
        return code
    
    def extract_code(self, text: str, lang='python'):
        """
        从LLM响应中提取代码块
        
        Args:
            text (str): LLM的完整响应文本
            lang (str): 编程语言标识，默认为'python'
            
        Returns:
            str: 提取出的代码内容，如果未找到代码块则返回原始文本
        """
        if text.startswith(f'```{lang}'):
            pattern = fr"```{lang}\n(.*?)\n```"
        elif text.startswith(f'{lang}'):
            pattern = fr"<{lang}>\n(.*?)\n</{lang}>"
        elif text.startswith(f'<{lang}>'):
            pattern = fr"<{lang}>\n(.*?)\n</{lang}>"
        else:
            return text

        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1)
        else:
            code = text
        return code