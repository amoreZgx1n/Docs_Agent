"""
集成测试脚本 - 测试文档处理流水线与意图处理器的集成
"""

import asyncio
import os
import sys
from pathlib import Path

# 设置环境变量
os.environ['SILICONFLOW_API_KEY'] = 'sk-lpylryqgkbbgxnxlarlrenupxzkfrqlrijyulgdamsegpugs'

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from selector import EnhancedIntentProcessor
import sentence_transformers


async def test_integration():
    """测试集成功能"""
    print("🧪 开始集成测试")
    print("=" * 50)
    
    try:
        # 初始化处理器
        print("📦 初始化增强意图处理器...")
        embedding_model = sentence_transformers.SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        processor = EnhancedIntentProcessor(embedding_model=embedding_model)
        print("✅ 处理器初始化成功")
        
        # 测试统计信息
        print("\n📊 测试统计信息获取...")
        stats = processor.get_statistics()
        print(f"数据库统计: {stats}")
        
        # 测试文档搜索（空数据库）
        print("\n🔍 测试文档搜索...")
        search_results = await processor.search_documents("测试查询", 3)
        print(f"搜索结果: {len(search_results)} 个")
        
        print("\n✅ 集成测试完成！")
        print("\n💡 下一步可以：")
        print("   1. 添加PDF或CSV文件到data/documents/目录")
        print("   2. 使用processor.process_query()处理文件")
        print("   3. 使用processor.search_documents()搜索文档")
        
    except Exception as e:
        print(f"❌ 集成测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_integration()) 