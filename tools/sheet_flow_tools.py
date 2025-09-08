"""
表格数据分析工具函数
实现 TABLE_ANALYSIS 任务所需的工具函数
"""

import sys
import os
from typing import Dict, List, Any, Optional
import pandas as pd

# 添加路径以导入相关模块
sys.path.append('/home/icesnls/workplace/zgx/work/Docs_Agent')
sys.path.append('/home/icesnls/workplace/zgx/work/Docs_Agent/src/llm_generate')

from llama_index.llms.zhipuai import ZhipuAI
from df_summary import DF_Summary


class InsightsGenerator:
    """业务洞察生成器"""

    def __init__(self, llm: Optional[ZhipuAI] = None):
        self.llm = llm or ZhipuAI(model="glm-4-flash", api_key=os.environ.get('ZHIPUAI_API_KEY', 'bdefa5d91470437380967660cf6592f0.UN5HkGKi6hYu8kVB'))

    def generate_insights(self, query: str, analysis_result: Dict[str, Any]) -> str:
        """
        生成业务洞察和建议

        Args:
            query: 用户查询
            analysis_result: data_analyze 函数的输出结果

        Returns:
            业务洞察和建议文本
        """
        try:
            # 提取分析结果中的关键信息
            markdown_content = analysis_result.get('markdown', '')
            meta_info = analysis_result.get('meta', {})

            # 构建洞察生成提示词
            prompt = self._build_insights_prompt(query, markdown_content, meta_info)

            # 调用 LLM 生成洞察
            response = self.llm.complete(prompt)
            insights = response.text.strip()

            # 格式化输出
            formatted_insights = self._format_insights(insights, meta_info)

            return formatted_insights

        except Exception as e:
            print(f"生成洞察时出错: {e}")
            return self._generate_fallback_insights(query, analysis_result)

    def _build_insights_prompt(self, query: str, markdown_content: str, meta_info: Dict[str, Any]) -> str:
        """构建洞察生成提示词"""

        tool_used = meta_info.get('tool', '未知')
        mapping_info = meta_info.get('mapping', {})

        prompt = f"""你是一个专业的数据分析师。请基于用户查询和数据分析结果，生成深入的业务洞察和 actionable 的建议。

用户查询：{query}

数据分析结果：
{markdown_content}

使用的分析工具：{tool_used}
字段映射：{mapping_info}

请从以下几个维度生成洞察：

1. **关键发现**：数据中最重要的模式、趋势或异常点
2. **业务含义**：这些发现对业务的实际意义
3. **发展趋势**：基于数据预测未来的发展趋势
4. **改进建议**：具体的、可执行的改进措施
5. **风险提醒**：潜在的风险点和注意事项

请用清晰的 Markdown 格式输出，确保洞察具有实际价值和可操作性。

输出格式：
## 📊 关键发现
[详细描述数据中的重要发现]

## 💼 业务含义
[解释这些发现对业务的意义]

## 📈 发展趋势
[基于数据预测未来趋势]

## 🎯 改进建议
[具体的改进措施和建议]

## ⚠️ 风险提醒
[潜在风险和注意事项]
"""

        return prompt

    def _format_insights(self, insights: str, meta_info: Dict[str, Any]) -> str:
        """格式化洞察输出"""

        # 添加元数据信息
        tool_info = meta_info.get('tool', '未知')
        formatted_output = f"""## 📊 数据分析洞察报告

**分析工具**: {tool_info}
**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

{insights}

---

> 此洞察基于数据分析结果自动生成，仅供参考。建议结合业务实际情况进行解读。
"""

        return formatted_output

    def _generate_fallback_insights(self, query: str, analysis_result: Dict[str, Any]) -> str:
        """生成兜底洞察（当LLM调用失败时）"""

        markdown_content = analysis_result.get('markdown', '')
        meta_info = analysis_result.get('meta', {})

        fallback_insights = f"""## 📊 数据分析洞察报告

**分析工具**: {meta_info.get('tool', '未知')}

### 🔍 基本分析结果
基于您的数据查询，已完成数据分析。以下是关键信息：

### 💡 初步洞察
- 数据分析已完成，包含可视化图表
- 建议查看图表了解数据分布和趋势
- 如需更深入的分析，请提供更具体的问题描述

### 🎯 建议下一步
1. 查看生成的可视化图表
2. 分析数据中的关键指标和趋势
3. 结合业务背景解读分析结果
4. 如有特定关注点，请进一步 уточ查询

---
> 此为自动生成的初步洞察，建议人工复核和解读。
"""

        return fallback_insights


# 全局实例
insights_generator = InsightsGenerator()


# 工具函数接口
def df_summary(dataframe: pd.DataFrame, head_num: int = 5) -> Dict[str, Any]:
    """
    生成数据框摘要统计

    Args:
        dataframe: 输入的数据框
        head_num: 显示的行数

    Returns:
        包含 head、info、describe 的摘要字典
    """
    summary_generator = DF_Summary(dataframe, head_num)
    return summary_generator.get_summary()


def data_analyze(user_query: str, dataframe: pd.DataFrame) -> Dict[str, Any]:
    """
    执行数据分析和可视化

    Args:
        user_query: 用户查询
        dataframe: 输入的数据框

    Returns:
        包含 markdown 和 meta 信息的分析报告字典
    """
    # 导入现有的 data_analyze 函数
    from data_analyze import data_analyze as original_data_analyze
    return original_data_analyze(user_query, dataframe)


def generate_insights(query: str, analysis_result: Dict[str, Any]) -> str:
    """
    生成业务洞察和建议

    Args:
        query: 用户查询
        analysis_result: data_analyze 函数的输出结果

    Returns:
        业务洞察和建议文本
    """
    return insights_generator.generate_insights(query, analysis_result)
