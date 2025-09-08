"""
任务执行器 (Actuator)
负责执行任务规划器生成的执行计划，调用相应的工具函数
"""

import sys
import os
import importlib
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import traceback

# 添加路径以导入相关模块
sys.path.append('/home/icesnls/workplace/zgx/work/Docs_Agent')
sys.path.append('/home/icesnls/workplace/zgx/work/Docs_Agent/src/llm_generate')
sys.path.append('/home/icesnls/workplace/zgx/work/Docs_Agent/tools')

from planner_v1 import TaskStep, ExecutionPlan


@dataclass
class StepResult:
    """步骤执行结果"""
    step_id: str
    tool_name: str
    success: bool
    output: Any
    error_message: Optional[str] = None
    execution_time: Optional[float] = None


class TaskActuator:
    """
    任务执行器
    负责执行任务规划器生成的执行计划
    """

    def __init__(self):
        self.step_results = {}  # 存储各步骤的执行结果
        self._load_tool_modules()

    def _load_tool_modules(self):
        """加载工具模块"""
        self.tool_modules = {}

        try:
            # 加载文档问答工具
            from tools.docs_flow_tools import (
                parse_document, chunk_text, semantic_search, generate_answer
            )
            self.tool_modules.update({
                'parse_document': parse_document,
                'chunk_text': chunk_text,
                'semantic_search': semantic_search,
                'generate_answer': generate_answer
            })
            print("文档问答工具加载成功")

        except ImportError as e:
            print(f"文档问答工具加载失败: {e}")

        try:
            # 加载表格分析工具
            from tools.sheet_flow_tools import df_summary, data_analyze, generate_insights
            self.tool_modules.update({
                'df_summary': df_summary,
                'data_analyze': data_analyze,
                'generate_insights': generate_insights
            })
            print("表格分析工具加载成功")

        except ImportError as e:
            print(f"表格分析工具加载失败: {e}")

        try:
            # 加载SQL分析工具
            from src.llm_generate.sql_generate import make_sql_generator
            from src.llm_generate.sql_executor import SQLExecutor
            from src.llm_generate.data_analyze import generate_sql_insights
            from src.llm_generate.llm import LLM
            from src.llm_generate.schema_manager_mysql import MySQLSchemaManager

            # 创建SQL工具实例
            self.sql_llm = LLM(backend="siliconflow")
            self.schema_manager = MySQLSchemaManager()
            self.sql_executor = SQLExecutor(engine=None)  # 需要在使用时传入真实的engine

            self.tool_modules.update({
                'sql_generate': self._sql_generate_wrapper,
                'sql_execute': self._sql_execute_wrapper,
                'generate_sql_insights': generate_sql_insights
            })
            print("SQL分析工具加载成功")

        except ImportError as e:
            print(f"SQL分析工具加载失败: {e}")

        print(f"共加载了 {len(self.tool_modules)} 个工具函数")

    def execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """
        执行完整的执行计划

        Args:
            plan: 执行计划

        Returns:
            包含所有步骤结果的字典
        """
        print(f"\n开始执行任务计划: {plan.task_type.value}")
        print(f"计划包含 {len(plan.steps)} 个步骤")

        self.step_results = {}  # 重置结果存储

        results = []
        success_count = 0

        for i, step in enumerate(plan.steps, 1):
            print(f"\n执行步骤 {i}/{len(plan.steps)}: {step.step_id} - {step.tool_name}")

            try:
                result = self._execute_step(step)
                results.append(result)

                if result.success:
                    success_count += 1
                    self.step_results[step.step_id] = result.output
                    print(f"✅ 步骤 {step.step_id} 执行成功")
                else:
                    print(f"❌ 步骤 {step.step_id} 执行失败: {result.error_message}")

            except Exception as e:
                error_msg = f"步骤执行异常: {str(e)}"
                print(f"💥 步骤 {step.step_id} 异常: {error_msg}")
                traceback.print_exc()

                result = StepResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    success=False,
                    output=None,
                    error_message=error_msg
                )
                results.append(result)

        # 生成执行摘要
        summary = {
            'total_steps': len(plan.steps),
            'successful_steps': success_count,
            'failed_steps': len(plan.steps) - success_count,
            'success_rate': success_count / len(plan.steps) if plan.steps else 0,
            'task_type': plan.task_type.value,
            'step_results': results,
            'final_output': self.step_results.get(plan.steps[-1].step_id) if plan.steps else None
        }

        print(f"\n📊 执行完成: {success_count}/{len(plan.steps)} 步骤成功")
        return summary

    def _execute_step(self, step: TaskStep) -> StepResult:
        """
        执行单个步骤

        Args:
            step: 任务步骤

        Returns:
            步骤执行结果
        """
        import time
        start_time = time.time()

        try:
            # 获取工具函数
            if step.tool_name not in self.tool_modules:
                raise ValueError(f"未找到工具函数: {step.tool_name}")

            tool_func = self.tool_modules[step.tool_name]

            # 处理参数中的步骤引用
            processed_params = self._process_parameters(step.parameters)

            # 执行工具函数
            output = tool_func(**processed_params)

            execution_time = time.time() - start_time

            return StepResult(
                step_id=step.step_id,
                tool_name=step.tool_name,
                success=True,
                output=output,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"

            return StepResult(
                step_id=step.step_id,
                tool_name=step.tool_name,
                success=False,
                output=None,
                error_message=error_msg,
                execution_time=execution_time
            )

    def _process_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理参数中的步骤引用

        Args:
            parameters: 原始参数字典

        Returns:
            处理后的参数字典
        """
        processed = {}

        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("from_step_"):
                # 处理步骤引用
                step_id = value
                if step_id in self.step_results:
                    processed[key] = self.step_results[step_id]
                else:
                    raise ValueError(f"引用的步骤结果不存在: {step_id}")
            else:
                processed[key] = value

        return processed

    def _sql_generate_wrapper(self, user_intent: str, db_config: Dict[str, Any], limit: int = 1000) -> str:
        """SQL生成包装器"""
        try:
            # 创建数据库引擎（这里需要根据实际配置创建）
            from sqlalchemy import create_engine

            # 从 db_config 创建数据库连接
            # 这里需要根据实际的数据库配置来创建连接字符串
            # 暂时使用示例配置
            db_url = f"mysql+pymysql://{db_config.get('user', 'root')}:{db_config.get('password', '')}@{db_config.get('host', 'localhost')}:{db_config.get('port', 3306)}/{db_config.get('database', 'test')}"

            engine = create_engine(db_url)
            generator = make_sql_generator(self.sql_llm, self.schema_manager)

            # 注意：这里需要 await，但当前是同步函数
            # 在实际使用中可能需要调整为异步
            import asyncio
            sql = asyncio.run(generator.generate_sql(user_intent, limit=limit))

            return sql

        except Exception as e:
            print(f"SQL生成失败: {e}")
            return f"SELECT * FROM {db_config.get('table', 'unknown_table')} LIMIT {limit};"

    def _sql_execute_wrapper(self, sql: str, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """SQL执行包装器"""
        try:
            from sqlalchemy import create_engine
            import pandas as pd

            # 从 db_config 创建数据库连接
            db_url = f"mysql+pymysql://{db_config.get('user', 'root')}:{db_config.get('password', '')}@{db_config.get('host', 'localhost')}:{db_config.get('port', 3306)}/{db_config.get('database', 'test')}"

            engine = create_engine(db_url)
            executor = SQLExecutor(engine)

            df, meta = executor.execute(sql)

            return {
                'dataframe': df,
                'meta': meta,
                'sql': sql
            }

        except Exception as e:
            print(f"SQL执行失败: {e}")
            # 返回空的DataFrame作为兜底
            empty_df = pd.DataFrame()
            return {
                'dataframe': empty_df,
                'meta': {'error': str(e)},
                'sql': sql
            }


# 全局执行器实例 - 延迟创建，避免导入时出错
_task_actuator = None


def _get_task_actuator() -> TaskActuator:
    """获取或创建任务执行器实例"""
    global _task_actuator
    if _task_actuator is None:
        _task_actuator = TaskActuator()
    return _task_actuator


def execute_plan(plan: ExecutionPlan) -> Dict[str, Any]:
    """
    执行执行计划的便捷函数

    Args:
        plan: 执行计划

    Returns:
        执行结果摘要
    """
    return _get_task_actuator().execute_plan(plan)


def get_step_result(step_id: str) -> Any:
    """
    获取指定步骤的执行结果

    Args:
        step_id: 步骤ID

    Returns:
        步骤执行结果
    """
    return _get_task_actuator().step_results.get(step_id)


if __name__ == "__main__":
    # 测试代码
    print("TaskActuator 测试")
    print(f"已加载的工具函数: {list(task_actuator.tool_modules.keys())}")
