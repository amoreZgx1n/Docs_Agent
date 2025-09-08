"""
Planner v1.0 - 简化版智能任务规划器

主要功能：
1. 意图识别：使用LLM识别用户意图（文档问答、表格数据分析、SQL数据分析）
2. 任务规划：根据意图生成具体的执行步骤
3. 记忆模块：集成memory.py的记忆管理功能

设计原则：
- 代码简洁，功能聚焦
- 模块化设计，职责清晰
- 集成记忆功能，提供上下文感知
"""

import os
import sys
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# 添加路径以导入相关模块
sys.path.append('/home/icesnls/workplace/zgx/work/Docs_Agent')
sys.path.append('/home/icesnls/workplace/zgx/work/Docs_Agent/src/llm_generate')

from memory import MemoryManager
from llama_index.llms.zhipuai import ZhipuAI


class TaskType(Enum):
    """任务类型枚举"""
    DOCUMENT_QA = "document_qa"          # 文档问答
    TABLE_ANALYSIS = "table_analysis"    # 表格数据分析
    SQL_ANALYSIS = "sql_analysis"        # SQL数据分析


@dataclass
class TaskStep:
    """任务步骤"""
    step_id: str
    tool_name: str
    parameters: Dict[str, Any]
    expected_output: str
    description: str


@dataclass
class ExecutionPlan:
    """执行计划"""
    task_type: TaskType
    steps: List[TaskStep]
    metadata: Dict[str, Any]


class IntentRecognizer:
    """意图识别器 - 使用LLM识别用户意图"""
    
    def __init__(self, llm: ZhipuAI, memory_manager: MemoryManager):
        self.llm = llm
        self.memory_manager = memory_manager
    
    def recognize_intent(self, query: str, data_input: Any) -> TaskType:
        """
        识别用户意图
        
        Args:
            query: 用户查询
            data_input: 输入数据（DataFrame、文档内容或数据库信息）
            
        Returns:
            TaskType: 识别出的任务类型
        """
        # 获取记忆上下文进行提示增强
        memory_context = self.memory_manager.get_relevant_context(query)
        
        # 构建增强提示词
        prompt = self._build_intent_prompt(query, data_input, memory_context)
        
        try:
            response = self.llm.complete(prompt)
            result = response.text.strip().lower()
            
            # 记录到记忆系统
            self.memory_manager.add_message('user', f"意图识别查询: {query}")
            self.memory_manager.add_message('assistant', f"识别结果: {result}")
            
            # 解析结果
            if "document_qa" in result:
                return TaskType.DOCUMENT_QA
            elif "sql_analysis" in result:
                return TaskType.SQL_ANALYSIS
            elif "table_analysis" in result:
                return TaskType.TABLE_ANALYSIS
            else:
                # 默认判断逻辑
                return self._fallback_intent_recognition(data_input)
                
        except Exception as e:
            print(f"LLM意图识别失败: {e}")
            return self._fallback_intent_recognition(data_input)
    
    def _build_intent_prompt(self, query: str, data_input: Any, memory_context: Dict) -> str:
        """构建意图识别的提示词"""
        
        # 分析输入数据类型
        if isinstance(data_input, pd.DataFrame):
            data_type = f"pandas DataFrame，形状: {data_input.shape}，列名: {list(data_input.columns)}"
        elif isinstance(data_input, dict) and 'db_config' in data_input:
            data_type = f"数据库连接信息，数据库: {data_input.get('database', '未知')}"
        else:
            data_type = f"文档内容，类型: {type(data_input).__name__}"
        
        # 构建记忆上下文
        memory_text = ""
        if memory_context['short_term']:
            recent_msgs = memory_context['short_term'][-3:]  # 最近3条消息
            memory_text = "最近对话历史:\n" + "\n".join([
                f"- {msg['role']}: {msg['content']}" for msg in recent_msgs
            ]) + "\n\n"
        
        prompt = f"""你是一个智能任务分类专家。根据用户查询和数据类型，判断应该执行哪种类型的任务。

{memory_text}用户查询: {query}

数据类型: {data_type}

请从以下三种任务类型中选择最合适的一种：

1. document_qa: 文档问答任务
   - 适用于：从文档中查找信息、回答问题、解释概念等
   - 输入特征：文档内容、文本数据

2. table_analysis: 表格数据分析任务  
   - 适用于：对DataFrame进行统计分析、可视化、数据探索等
   - 输入特征：pandas DataFrame、结构化数据

3. sql_analysis: SQL数据分析任务
   - 适用于：需要从数据库查询数据进行分析
   - 输入特征：数据库连接信息、需要复杂查询的分析需求

请只回答任务类型名称，如: "table_analysis"
"""
        return prompt
    
    def _fallback_intent_recognition(self, data_input: Any) -> TaskType:
        """降级意图识别逻辑"""
        if isinstance(data_input, pd.DataFrame):
            return TaskType.TABLE_ANALYSIS
        elif isinstance(data_input, dict) and 'db_config' in data_input:
            return TaskType.SQL_ANALYSIS
        else:
            return TaskType.DOCUMENT_QA


class TaskPlanner:
    """任务规划器 - 根据意图生成执行计划"""
    
    def __init__(self, llm: ZhipuAI, memory_manager: MemoryManager):
        self.llm = llm
        self.memory_manager = memory_manager
    
    def create_plan(self, task_type: TaskType, query: str, data_input: Any) -> ExecutionPlan:
        """
        创建执行计划
        
        Args:
            task_type: 任务类型
            query: 用户查询
            data_input: 输入数据
            
        Returns:
            ExecutionPlan: 执行计划
        """
        # 获取记忆上下文
        memory_context = self.memory_manager.get_relevant_context(query)
        
        if task_type == TaskType.DOCUMENT_QA:
            plan = self._create_document_qa_plan(query, data_input, memory_context)
        elif task_type == TaskType.TABLE_ANALYSIS:
            plan = self._create_table_analysis_plan(query, data_input, memory_context)
        elif task_type == TaskType.SQL_ANALYSIS:
            plan = self._create_sql_analysis_plan(query, data_input, memory_context)
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")
        
        # 记录规划过程到记忆
        self.memory_manager.add_message('user', f"任务规划请求: {query}")
        self.memory_manager.add_message('assistant', f"生成了{len(plan.steps)}步执行计划")
        
        return plan
    
    def _create_document_qa_plan(self, query: str, document: Any, memory_context: Dict) -> ExecutionPlan:
        """创建文档问答执行计划"""
        steps = [
            TaskStep(
                step_id="step_001",
                tool_name="parse_document",
                parameters={"content": document},
                expected_output="解析后的文档结构",
                description="解析文档内容"
            ),
            TaskStep(
                step_id="step_002", 
                tool_name="chunk_text",
                parameters={"chunk_size": 500, "overlap": 50},
                expected_output="文档分块列表",
                description="将文档分割为文本块"
            ),
            TaskStep(
                step_id="step_003",
                tool_name="semantic_search",
                parameters={"query": query, "top_k": 3},
                expected_output="相关文本片段列表",
                description="搜索相关文本片段"
            ),
            TaskStep(
                step_id="step_004",
                tool_name="generate_answer",
                parameters={"query": query, "context": "from_step_003"},
                expected_output="生成的答案文本",
                description="基于上下文生成答案"
            )
        ]
        
        return ExecutionPlan(
            task_type=TaskType.DOCUMENT_QA,
            steps=steps,
            metadata={"query": query, "memory_context": memory_context}
        )
    
    def _create_table_analysis_plan(self, query: str, df: pd.DataFrame, memory_context: Dict) -> ExecutionPlan:
        """创建表格数据分析执行计划"""
        steps = [
            TaskStep(
                step_id="step_001",
                tool_name="df_summary",
                parameters={"dataframe": df, "head_num": 5},
                expected_output="数据摘要字典包含head、info、describe",
                description="生成数据框摘要统计"
            ),
            TaskStep(
                step_id="step_002",
                tool_name="data_analyze",
                parameters={"user_query": query, "dataframe": df},
                expected_output="分析报告字典包含markdown和meta信息",
                description="执行数据分析和可视化"
            ),
            TaskStep(
                step_id="step_003",
                tool_name="generate_insights",
                parameters={"query": query, "analysis_result": "from_step_002"},
                expected_output="业务洞察和建议文本",
                description="生成业务洞察和建议"
            )
        ]
        
        return ExecutionPlan(
            task_type=TaskType.TABLE_ANALYSIS,
            steps=steps,
            metadata={
                "query": query, 
                "df_shape": df.shape,
                "df_columns": list(df.columns),
                "memory_context": memory_context
            }
        )
    
    def _create_sql_analysis_plan(self, query: str, db_info: Dict, memory_context: Dict) -> ExecutionPlan:
        """创建SQL数据分析执行计划"""
        steps = [
            TaskStep(
                step_id="step_001",
                tool_name="sql_generate",
                parameters={
                    "user_intent": query,
                    "db_config": db_info,
                    "limit": 1000
                },
                expected_output="生成的SQL查询语句",
                description="生成SQL查询语句"
            ),
            TaskStep(
                step_id="step_002",
                tool_name="sql_execute",
                parameters={
                    "sql": "from_step_001",
                    "db_config": db_info
                },
                expected_output="查询结果DataFrame和元数据",
                description="执行SQL查询"
            ),
            TaskStep(
                step_id="step_003",
                tool_name="data_analyze",
                parameters={
                    "user_query": query,
                    "dataframe": "from_step_002"
                },
                expected_output="分析报告字典",
                description="对查询结果进行数据分析"
            ),
            TaskStep(
                step_id="step_004",
                tool_name="generate_sql_insights",
                parameters={
                    "query": query,
                    "sql": "from_step_001", 
                    "analysis_result": "from_step_003"
                },
                expected_output="SQL分析洞察报告",
                description="生成SQL分析洞察"
            )
        ]
        
        return ExecutionPlan(
            task_type=TaskType.SQL_ANALYSIS,
            steps=steps,
            metadata={
                "query": query,
                "database": db_info.get('database', '未知'),
                "memory_context": memory_context
            }
        )


class PlannerV1:
    """
    简化版智能任务规划器
    
    集成三大功能模块：
    1. 意图识别
    2. 任务规划  
    3. 记忆管理
    """
    
    def __init__(self, api_key: str = None):
        # 初始化LLM
        if not api_key:
            api_key = os.environ.get('ZHIPUAI_API_KEY', 'bdefa5d91470437380967660cf6592f0.UN5HkGKi6hYu8kVB')
        
        self.llm = ZhipuAI(model="glm-4-flash", api_key=api_key)
        
        # 初始化记忆管理器
        self.memory_manager = MemoryManager()
        
        # 初始化子模块
        self.intent_recognizer = IntentRecognizer(self.llm, self.memory_manager)
        self.task_planner = TaskPlanner(self.llm, self.memory_manager)
        
        print("PlannerV1 初始化完成")
        print(f"- LLM: {self.llm.model}")
        print(f"- 记忆管理器已启用")
    
    def plan(self, query: str, data_input: Any) -> ExecutionPlan:
        """
        主要接口：根据用户查询和输入数据生成执行计划
        
        Args:
            query: 用户查询
            data_input: 输入数据
            
        Returns:
            ExecutionPlan: 执行计划
        """
        print(f"\n开始规划任务: {query}")
        
        # 1. 意图识别
        task_type = self.intent_recognizer.recognize_intent(query, data_input)
        print(f"识别任务类型: {task_type.value}")
        
        # 2. 任务规划
        plan = self.task_planner.create_plan(task_type, query, data_input)
        print(f"生成执行计划: {len(plan.steps)} 个步骤")
        
        return plan
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆系统摘要"""
        return self.memory_manager.get_memory_summary()
    
    def clear_memory(self, memory_type: str = "short"):
        """清空记忆"""
        self.memory_manager.clear_memory(memory_type)


def test_planner_v1():
    """测试函数"""
    print("=" * 60)
    print("PlannerV1 测试")
    print("=" * 60)
    
    # 初始化规划器
    planner = PlannerV1()
    
    # 测试1: 表格数据分析
    print("\n【测试1: 表格数据分析】")
    df_test = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100),
        'sales': range(100, 200),
        'region': ['北京', '上海'] * 50,
        'profit': [x * 0.2 for x in range(100, 200)]
    })
    
    plan1 = planner.plan("分析销售趋势，生成可视化图表", df_test)
    
    print(f"任务类型: {plan1.task_type.value}")
    print("执行步骤:")
    for step in plan1.steps:
        print(f"  {step.step_id}: {step.tool_name}")
        print(f"    描述: {step.description}")
        print(f"    期望输出: {step.expected_output}")
        print()
    
    # 测试2: SQL数据分析
    print("\n【测试2: SQL数据分析】")
    db_info = {
        'db_config': {
            'host': 'localhost',
            'database': 'sales_db',
            'user': 'test_user'
        },
        'database': 'sales_db'
    }
    
    plan2 = planner.plan("查询最近30天的销售数据并进行趋势分析", db_info)
    
    print(f"任务类型: {plan2.task_type.value}")
    print("执行步骤:")
    for step in plan2.steps:
        print(f"  {step.step_id}: {step.tool_name}")
        print(f"    描述: {step.description}")
        print(f"    期望输出: {step.expected_output}")
        print()
    
    # 测试3: 文档问答
    print("\n【测试3: 文档问答】")
    document = """
    公司政策文档
    
    休假政策：
    1. 年假：员工每年享有15天带薪年假
    2. 病假：员工每年享有10天带薪病假
    3. 事假：员工可申请无薪事假，需提前3天申请
    
    薪资政策：
    1. 基本工资按月发放
    2. 绩效奖金按季度发放
    3. 年终奖金根据公司业绩确定
    """
    
    plan3 = planner.plan("公司的休假政策是什么？", document)
    
    print(f"任务类型: {plan3.task_type.value}")
    print("执行步骤:")
    for step in plan3.steps:
        print(f"  {step.step_id}: {step.tool_name}")
        print(f"    描述: {step.description}")
        print(f"    期望输出: {step.expected_output}")
        print()
    
    # 测试记忆功能
    print("\n【记忆系统摘要】")
    memory_summary = planner.get_memory_summary()
    print(f"短期记忆: {memory_summary['short_term_count']} 条")
    print(f"长期记忆: {memory_summary['long_term_count']} 条")
    print(f"总消息数: {memory_summary['stats']['total_messages']}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_planner_v1()

