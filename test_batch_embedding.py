"""
测试批量向量化性能的脚本
"""

import asyncio
import time
import os
import sys
from pathlib import Path

# 设置环境变量
os.environ['SILICONFLOW_API_KEY'] = 'sk-lpylryqgkbbgxnxlarlrenupxzkfrqlrijyulgdamsegpugs'

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline.document_pipeline import DocumentPipeline
from src.utils.config import get_config
from src.utils.logger import setup_logger


async def test_batch_embedding_performance():
    """测试批量向量化性能"""
    print("🧪 开始测试批量向量化性能")
    print("=" * 60)
    
    # 初始化日志和配置
    setup_logger()
    config = get_config()
    
    # 初始化文档处理流水线
    pipeline = DocumentPipeline(config)
    
    # 测试文件路径
    pdf_file = Path("data/documents/毛选.pdf")
    
    if not pdf_file.exists():
        print(f"❌ 测试文件不存在: {pdf_file}")
        return
    
    print(f"📄 测试文件: {pdf_file}")
    print(f"📊 文件大小: {pdf_file.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        # 记录开始时间
        start_time = time.time()
        
        print("\n🚀 开始处理PDF文档...")
        result = pipeline.process_document(pdf_file)
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        if result.success:
            print(f"\n✅ 处理成功!")
            print(f"📊 处理统计:")
            print(f"   - 文本块数量: {result.chunks_count}")
            print(f"   - 总耗时: {total_time:.2f} 秒")
            print(f"   - 平均每块耗时: {total_time/result.chunks_count:.3f} 秒")
            print(f"   - 处理速度: {result.chunks_count/total_time:.2f} 块/秒")
            
            # 获取向量数据库统计信息
            stats = pipeline.get_statistics()
            print(f"\n🗄️ 向量数据库统计:")
            print(f"   - 总文档数: {stats.get('total_documents', 0)}")
            print(f"   - 集合数量: {stats.get('collections_count', 0)}")
            
            # 测试搜索功能
            print(f"\n🔍 测试搜索功能...")
            search_start = time.time()
            search_results = pipeline.search_documents("毛泽东思想", n_results=3)
            search_time = time.time() - search_start
            
            print(f"   - 搜索耗时: {search_time:.3f} 秒")
            print(f"   - 搜索结果数: {len(search_results)}")
            
            if search_results:
                print(f"   - 第一个结果相似度: {search_results[0].get('score', 0):.4f}")
            
        else:
            print(f"\n❌ 处理失败: {result.error_message}")
            
    except Exception as e:
        print(f"\n❌ 测试过程中发生异常: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_embedding_cache():
    """测试向量化缓存功能"""
    print("\n" + "=" * 60)
    print("🧪 测试向量化缓存功能")
    print("=" * 60)
    
    from src.text_processor.text_embedder import TextEmbedder
    
    # 初始化向量化器
    embedder = TextEmbedder()
    
    # 测试文本
    test_texts = [
        "这是第一个测试文本，用于测试向量化功能。",
        "这是第二个测试文本，内容略有不同。",
        "这是第三个测试文本，用于验证批量处理。"
    ]
    
    print(f"📝 测试文本数量: {len(test_texts)}")
    
    # 第一次向量化（应该生成新向量）
    print("\n🔄 第一次向量化（生成新向量）...")
    start_time = time.time()
    embeddings1 = embedder.embed_texts(test_texts)
    time1 = time.time() - start_time
    print(f"   耗时: {time1:.3f} 秒")
    
    # 第二次向量化（应该使用缓存）
    print("\n⚡ 第二次向量化（使用缓存）...")
    start_time = time.time()
    embeddings2 = embedder.embed_texts(test_texts)
    time2 = time.time() - start_time
    print(f"   耗时: {time2:.3f} 秒")
    
    # 计算性能提升
    if time1 > 0:
        speedup = time1 / time2
        print(f"\n📈 缓存性能提升: {speedup:.2f}x")
        print(f"   时间节省: {((time1 - time2) / time1 * 100):.1f}%")
    
    # 获取缓存信息
    cache_info = embedder.get_cache_info()
    print(f"\n🗄️ 缓存信息:")
    print(f"   - 缓存文件数: {cache_info.get('file_count', 0)}")
    print(f"   - 缓存大小: {cache_info.get('total_size_mb', 0):.2f} MB")


if __name__ == "__main__":
    asyncio.run(test_batch_embedding_performance())
    asyncio.run(test_embedding_cache()) 