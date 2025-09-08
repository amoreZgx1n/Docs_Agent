"""
Docs_Agent 主程序
完整的智能文档分析Agent流程：文件解析 -> 任务规划 -> 执行器 -> 结果输出
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings

# 添加项目路径
sys.path.append('/home/icesnls/workplace/zgx/work/Docs_Agent')

# 导入核心模块
from file_parser import (
    parse_document_input, parse_sheet_input, parse_sql_input,
    DocumentInput, SheetInput, SQLInput
)
from planner_v1 import PlannerV1, TaskType
from actuator import execute_plan
from memory import MemoryManager

# 忽略警告以获得更清洁的输出
warnings.filterwarnings('ignore')


@dataclass
class AgentResult:
    """Agent执行结果"""
    task_type: TaskType
    input_info: Dict[str, Any]
    plan: Any
    execution_result: Dict[str, Any]
    final_output: Any
    execution_time: float
    status: str


class DocsAgent:
    """
    智能文档分析Agent
    集成文件解析、任务规划和执行器的完整流程
    """

    def __init__(self, llm_api_key: Optional[str] = None):
        print("初始化 DocsAgent...")

        # 初始化核心组件
        self.planner = PlannerV1(api_key=llm_api_key)
        self.memory_manager = MemoryManager()

        # 确保测试数据目录存在
        self.test_data_dir = Path("/home/icesnls/workplace/zgx/work/Docs_Agent/test_data")
        self.test_data_dir.mkdir(exist_ok=True)

        print("DocsAgent 初始化完成")
        print(f"测试数据目录: {self.test_data_dir}")

    def create_test_files(self):
        """创建测试文件"""
        print("\n创建测试文件...")

        # 创建测试PDF文件
        self._create_test_pdf()

        # 创建测试Excel文件
        self._create_test_excel()

        # 创建测试数据库配置
        self._create_test_db_config()

        print("测试文件创建完成")

    def _create_test_pdf(self):
        """创建测试PDF文件"""
        pdf_path = self.test_data_dir / "company_policy.pdf"

        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont

            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            width, height = letter

            # 设置字体
            c.setFont("Helvetica-Bold", 16)
            c.drawString(100, height - 100, "公司政策文档")

            c.setFont("Helvetica-Bold", 14)
            c.drawString(100, height - 140, "休假政策：")

            c.setFont("Helvetica", 12)
            c.drawString(120, height - 170, "1. 年假：员工每年享有15天带薪年假")
            c.drawString(120, height - 190, "2. 病假：员工每年享有10天带薪病假")
            c.drawString(120, height - 210, "3. 事假：员工可申请无薪事假，需提前3天申请")

            c.setFont("Helvetica-Bold", 14)
            c.drawString(100, height - 250, "薪资政策：")

            c.setFont("Helvetica", 12)
            c.drawString(120, height - 280, "1. 基本工资按月发放")
            c.drawString(120, height - 300, "2. 绩效奖金按季度发放")
            c.drawString(120, height - 320, "3. 年终奖金根据公司业绩确定")

            c.setFont("Helvetica-Bold", 14)
            c.drawString(100, height - 360, "工作时间：")

            c.setFont("Helvetica", 12)
            c.drawString(120, height - 390, "1. 标准工作时间：每周5天，每天8小时")
            c.drawString(120, height - 410, "2. 弹性工作制：允许在一定范围内调整工作时间")
            c.drawString(120, height - 430, "3. 加班补偿：按国家规定执行")

            c.save()
            print(f"测试PDF文件创建成功: {pdf_path}")

        except ImportError:
            print("reportlab 未安装，使用文本文件替代")
            # 创建文本文件作为替代
            txt_content = """公司政策文档

休假政策：
1. 年假：员工每年享有15天带薪年假
2. 病假：员工每年享有10天带薪病假
3. 事假：员工可申请无薪事假，需提前3天申请

薪资政策：
1. 基本工资按月发放
2. 绩效奖金按季度发放
3. 年终奖金根据公司业绩确定

工作时间：
1. 标准工作时间：每周5天，每天8小时
2. 弹性工作制：允许在一定范围内调整工作时间
3. 加班补偿：按国家规定执行
"""
            txt_path = self.test_data_dir / "company_policy.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(txt_content)
            print(f"测试文本文件创建成功: {txt_path}")

    def _create_test_excel(self):
        """创建测试Excel文件"""
        excel_path = self.test_data_dir / "sales_data.xlsx"

        # 创建销售数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sales_data = {
            '日期': dates,
            '销售额': [100 + i * 2 + (i % 10) * 10 for i in range(100)],
            '区域': ['北京', '上海', '广州', '深圳', '杭州'] * 20,
            '利润': [sale * 0.15 + (i % 5) * 5 for i, sale in enumerate([100 + i * 2 + (i % 10) * 10 for i in range(100)])],
            '客户数': [10 + i % 20 for i in range(100)],
            '产品类别': ['电子产品', '服装', '食品', '家居', '图书'] * 20
        }

        df = pd.DataFrame(sales_data)

        # 保存为Excel文件
        df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"测试Excel文件创建成功: {excel_path}")
        print(f"   数据形状: {df.shape}")
        print(f"   列名: {list(df.columns)}")

    def _create_test_db_config(self):
        """创建测试数据库配置"""
        db_config_path = self.test_data_dir / "db_config.json"

        db_config = {
            "host": "localhost",
            "port": 3306,
            "database": "sales_db",
            "user": "test_user",
            "password": "test_password",
            "charset": "utf8mb4",
            "table": "monthly_sales"
        }

        with open(db_config_path, 'w', encoding='utf-8') as f:
            json.dump(db_config, f, indent=2, ensure_ascii=False)

        print(f"测试数据库配置创建成功: {db_config_path}")

        # 创建模拟的数据库模式信息
        schema_path = self.test_data_dir / "db_schema.json"
        schema_info = {
            "tables": [
                {
                    "name": "monthly_sales",
                    "columns": [
                        {"name": "id", "type": "INT", "nullable": False, "primary_key": True},
                        {"name": "month", "type": "VARCHAR(10)", "nullable": False},
                        {"name": "sales_amount", "type": "DECIMAL(10,2)", "nullable": False},
                        {"name": "region", "type": "VARCHAR(50)", "nullable": False},
                        {"name": "product_category", "type": "VARCHAR(100)", "nullable": False},
                        {"name": "customer_count", "type": "INT", "nullable": False}
                    ],
                    "row_count_estimate": 1200
                }
            ]
        }

        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(schema_info, f, indent=2, ensure_ascii=False)

        print(f"测试数据库模式信息创建成功: {schema_path}")

    def process_document_qa(self, query: str, file_path: str) -> AgentResult:
        """
        处理文档问答任务

        Args:
            query: 用户查询
            file_path: 文档文件路径

        Returns:
            AgentResult: 处理结果
        """
        print(f"\n 开始处理文档问答任务")
        print(f"   查询: {query}")
        print(f"   文件: {file_path}")

        start_time = datetime.now()

        try:
            # 步骤1: 文件解析
            print("    步骤1: 解析输入文件...")
            doc_input = parse_document_input(query, file_path)
            print("    文件解析完成"            )
            print(f"      内容长度: {len(doc_input.content)} 字符")
            print(f"      页数: {doc_input.metadata.get('page_count', '未知')}")

            # 步骤2: 任务规划
            print("    步骤2: 生成执行计划...")
            plan = self.planner.plan(doc_input.query, doc_input.content)
            print("    任务规划完成"            )
            print(f"      任务类型: {plan.task_type.value}")
            print(f"      执行步骤数: {len(plan.steps)}")

            # 步骤3: 执行计划
            print("     步骤3: 执行任务计划...")
            execution_result = execute_plan(plan)
            print("    任务执行完成"            )
            print(f"      成功步骤: {execution_result['successful_steps']}/{execution_result['total_steps']}")

            # 计算执行时间
            execution_time = (datetime.now() - start_time).total_seconds()

            return AgentResult(
                task_type=TaskType.DOCUMENT_QA,
                input_info={
                    'query': query,
                    'file_path': file_path,
                    'file_type': 'document',
                    'content_length': len(doc_input.content)
                },
                plan=plan,
                execution_result=execution_result,
                final_output=execution_result.get('final_output'),
                execution_time=execution_time,
                status='success'
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            print(f"   处理失败: {e}")

            return AgentResult(
                task_type=TaskType.DOCUMENT_QA,
                input_info={'query': query, 'file_path': file_path},
                plan=None,
                execution_result={},
                final_output=None,
                execution_time=execution_time,
                status='failed'
            )

    def process_table_analysis(self, query: str, file_path: str) -> AgentResult:
        """
        处理表格分析任务

        Args:
            query: 用户查询
            file_path: 表格文件路径

        Returns:
            AgentResult: 处理结果
        """
        print(f"\n开始处理表格分析任务")
        print(f"   查询: {query}")
        print(f"   文件: {file_path}")

        start_time = datetime.now()

        try:
            # 步骤1: 文件解析
            print("   步骤1: 解析输入文件...")
            sheet_input = parse_sheet_input(query, file_path)
            print("   文件解析完成"            )
            print(f"      数据形状: {sheet_input.dataframe.shape}")
            print(f"      列名: {sheet_input.metadata.get('columns', [])}")

            # 步骤2: 任务规划
            print("   步骤2: 生成执行计划...")
            plan = self.planner.plan(sheet_input.query, sheet_input.dataframe)
            print("   任务规划完成"            )
            print(f"      任务类型: {plan.task_type.value}")
            print(f"      执行步骤数: {len(plan.steps)}")

            # 步骤3: 执行计划
            print("   步骤3: 执行任务计划...")
            execution_result = execute_plan(plan)
            print("   任务执行完成"            )
            print(f"      成功步骤: {execution_result['successful_steps']}/{execution_result['total_steps']}")

            # 计算执行时间
            execution_time = (datetime.now() - start_time).total_seconds()

            return AgentResult(
                task_type=TaskType.TABLE_ANALYSIS,
                input_info={
                    'query': query,
                    'file_path': file_path,
                    'file_type': 'sheet',
                    'data_shape': sheet_input.dataframe.shape,
                    'columns': list(sheet_input.dataframe.columns)
                },
                plan=plan,
                execution_result=execution_result,
                final_output=execution_result.get('final_output'),
                execution_time=execution_time,
                status='success'
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            print(f"   处理失败: {e}")

            return AgentResult(
                task_type=TaskType.TABLE_ANALYSIS,
                input_info={'query': query, 'file_path': file_path},
                plan=None,
                execution_result={},
                final_output=None,
                execution_time=execution_time,
                status='failed'
            )

    def process_sql_analysis(self, query: str, db_config_path: str) -> AgentResult:
        """
        处理SQL分析任务

        Args:
            query: 用户查询
            db_config_path: 数据库配置文件路径

        Returns:
            AgentResult: 处理结果
        """
        print(f"\n开始处理SQL分析任务")
        print(f"   查询: {query}")
        print(f"   配置: {db_config_path}")

        start_time = datetime.now()

        try:
            # 步骤1: 解析数据库配置
            print("   步骤1: 解析数据库配置...")
            sql_input = parse_sql_input(query, db_config_path)
            print("   配置解析完成"            )
            print(f"      数据库: {sql_input.db_config.get('database')}")
            print(f"      目标表: {sql_input.table_info.get('table_name') if sql_input.table_info else '未指定'}")

            # 步骤2: 任务规划
            print("   步骤2: 生成执行计划...")
            plan = self.planner.plan(sql_input.query, sql_input.db_config)
            print("   任务规划完成"            )
            print(f"      任务类型: {plan.task_type.value}")
            print(f"      执行步骤数: {len(plan.steps)}")

            # 步骤3: 执行计划（注意：SQL执行需要真实的数据库连接，这里会模拟）
            print("   步骤3: 执行任务计划...")
            print("   注意: SQL执行需要真实的数据库连接，这里将跳过实际执行")
            execution_result = {
                'total_steps': len(plan.steps),
                'successful_steps': len(plan.steps),  # 模拟全部成功
                'failed_steps': 0,
                'success_rate': 1.0,
                'task_type': plan.task_type.value,
                'step_results': [],
                'simulation_note': 'SQL执行已模拟，实际使用需要配置真实数据库'
            }

            # 计算执行时间
            execution_time = (datetime.now() - start_time).total_seconds()

            return AgentResult(
                task_type=TaskType.SQL_ANALYSIS,
                input_info={
                    'query': query,
                    'db_config_path': db_config_path,
                    'file_type': 'database',
                    'database': sql_input.db_config.get('database'),
                    'table': sql_input.table_info.get('table_name') if sql_input.table_info else None
                },
                plan=plan,
                execution_result=execution_result,
                final_output="SQL分析模拟完成 - 实际执行需要配置数据库连接",
                execution_time=execution_time,
                status='success'
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            print(f"   处理失败: {e}")

            return AgentResult(
                task_type=TaskType.SQL_ANALYSIS,
                input_info={'query': query, 'db_config_path': db_config_path},
                plan=None,
                execution_result={},
                final_output=None,
                execution_time=execution_time,
                status='failed'
            )

    def display_result(self, result: AgentResult):
        """展示处理结果"""
        print(f"\n{'='*80}")
        print(f"任务处理结果")
        print(f"{'='*80}")

        print(f"任务类型: {result.task_type.value}")
        print(f"执行时间: {result.execution_time:.2f}秒")
        print(f"状态: {'成功' if result.status == 'success' else '失败'}")

        print(f"\n输入信息:")
        for key, value in result.input_info.items():
            print(f"   {key}: {value}")

        if result.plan:
            print(f"\n执行计划:")
            print(f"   总步骤数: {len(result.plan.steps)}")
            for i, step in enumerate(result.plan.steps, 1):
                print(f"   {i}. {step.step_id}: {step.tool_name}")
                print(f"      描述: {step.description}")

        if result.execution_result:
            print(f"\n执行结果:")
            if 'successful_steps' in result.execution_result:
                success_rate = result.execution_result.get('success_rate', 0) * 100
                print(f"   成功率: {success_rate:.1f}%")
                print(f"   成功步骤: {result.execution_result['successful_steps']}")
                print(f"   总步骤数: {result.execution_result['total_steps']}")

            if 'simulation_note' in result.execution_result:
                print(f"  模拟说明: {result.execution_result['simulation_note']}")

        if result.final_output:
            print(f"\n最终输出:")
            if isinstance(result.final_output, str):
                # 限制输出长度
                output_preview = result.final_output[:500] + "..." if len(result.final_output) > 500 else result.final_output
                print(f"   {output_preview}")
            else:
                print(f"   {type(result.final_output).__name__}: {result.final_output}")

        print(f"{'='*80}\n")

    def run_demo(self):
        """运行完整演示"""
        print("DocsAgent 完整演示")
        print("=" * 80)

        # 创建测试文件
        self.create_test_files()

        results = []

        # 演示1: 文档问答
        print("\n" + "="*80)
        print("演示1: 文档问答任务")
        print("="*80)

        pdf_path = self.test_data_dir / "company_policy.pdf"
        txt_path = self.test_data_dir / "company_policy.txt"

        # 优先使用PDF，如果没有则使用文本文件
        doc_path = str(pdf_path) if pdf_path.exists() else str(txt_path)

        doc_result = self.process_document_qa(
            "公司的休假政策是什么？",
            doc_path
        )
        results.append(doc_result)
        self.display_result(doc_result)

        # 演示2: 表格分析
        print("\n" + "="*80)
        print("演示2: 表格分析任务")
        print("="*80)

        excel_path = self.test_data_dir / "sales_data.xlsx"
        sheet_result = self.process_table_analysis(
            "分析销售趋势，找出最畅销的产品类别",
            str(excel_path)
        )
        results.append(sheet_result)
        self.display_result(sheet_result)

        # 演示3: SQL分析
        print("\n" + "="*80)
        print("演示3: SQL分析任务")
        print("="*80)

        db_config_path = self.test_data_dir / "db_config.json"
        sql_result = self.process_sql_analysis(
            "查询最近6个月的销售数据，按区域统计销售额",
            str(db_config_path)
        )
        results.append(sql_result)
        self.display_result(sql_result)

        # 总结报告
        self._print_summary_report(results)

    def _print_summary_report(self, results: list):
        """打印总结报告"""
        print("\n" + "="*80)
        print("处理总结报告")
        print("="*80)

        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.status == 'success')
        total_time = sum(r.execution_time for r in results)

        print(f"总体统计:")
        print(f"   总任务数: {total_tasks}")
        print(f"   成功任务: {successful_tasks}")
        print(f"   失败任务: {total_tasks - successful_tasks}")
        print(f"   总耗时: {total_time:.2f}秒")
        print(f"\n详细结果:")
        for i, result in enumerate(results, 1):
            status_text = "成功" if result.status == 'success' else "失败"
            print(f"   {i}. {status_text} - {result.task_type.value}"xoexiu负)
        print(f"\n测试文件位置: {self.test_data_dir}")
        print(f"   - 文档: company_policy.pdf/txt")
        print(f"   - 表格: sales_data.xlsx")
        print(f"   - 数据库配置: db_config.json")

        print("\n演示完成！")
        print("="*80)


def main():
    """主函数"""
    try:
        # 初始化Agent
        agent = DocsAgent()

        # 运行演示
        agent.run_demo()

    except KeyboardInterrupt:
        print("\n\n用户中断演示")
    except Exception as e:
        print(f"\n演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
