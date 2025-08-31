#!/usr/bin/env python3
"""
文档问答测试
验证PDF处理和搜索功能
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline.document_pipeline import DocumentPipeline
from src.utils.config import get_config


def test_document_qa():
    """测试文档问答功能"""
    print("🧪 测试文档问答功能")
    print("=" * 50)
    
    # 获取配置
    config = get_config()
    
    # 初始化文档处理流水线
    pipeline = DocumentPipeline(config)
    
    # 测试PDF文件路径
    pdf_path = "data/documents/毛选.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF文件不存在: {pdf_path}")
        return False
    
    print(f"📄 处理PDF文件: {pdf_path}")
    
    # 处理PDF文档
    try:
        result = pipeline.process_document(pdf_path)
        
        if result.success:
            print(f"✅ PDF处理成功！")
            print(f"📊 生成文本块数量: {result.chunks_count}")
            print(f"⏱️  处理时间: {result.processing_time:.2f}秒")
            
            # 测试搜索功能
            print("\n🔍 测试搜索功能...")
            
            # 测试几个不同的查询
            test_queries = [
                "什么是毛泽东思想？",
                "抗日战争的主要特点是什么？",
                "新民主主义革命的基本纲领",
                "人民战争的特点",
                "统一战线的重要性"
            ]
            
            for query in test_queries:
                print(f"\n❓ 查询: {query}")
                
                try:
                    search_results = pipeline.search_documents(query, n_results=3)
                    
                    if search_results:
                        print(f"✅ 找到 {len(search_results)} 个相关结果:")
                        for i, result in enumerate(search_results, 1):
                            print(f"  {i}. 相似度: {result.get('distance', 'N/A'):.4f}")
                            print(f"     来源: {result.get('metadata', {}).get('source', 'N/A')}")
                            print(f"     内容: {result.get('text', '')[:100]}...")
                    else:
                        print("❌ 未找到相关结果")
                        
                except Exception as e:
                    print(f"❌ 搜索失败: {e}")
            
            # 测试统计功能
            print("\n📈 测试统计功能...")
            try:
                stats = pipeline.get_statistics()
                print(f"✅ 数据库统计:")
                print(f"   总文档数: {stats.get('total_documents', 'N/A')}")
                print(f"   存储路径: {stats.get('storage_path', 'N/A')}")
                print(f"   存储大小: {stats.get('storage_size_mb', 'N/A'):.2f}MB")
                print(f"   向量维度: {stats.get('vector_dimension', 'N/A')}")
                
            except Exception as e:
                print(f"❌ 统计失败: {e}")
            
            return True
            
        else:
            print(f"❌ PDF处理失败: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_interactive_qa():
    """交互式问答测试"""
    print("\n💬 交互式问答测试")
    print("=" * 50)
    print("输入 'quit' 退出，输入 'stats' 查看统计信息")
    
    # 获取配置
    config = get_config()
    
    # 初始化文档处理流水线
    pipeline = DocumentPipeline(config)
    
    while True:
        try:
            query = input("\n❓ 请输入您的问题: ").strip()
            
            if query.lower() == 'quit':
                print("👋 再见！")
                break
                
            if query.lower() == 'stats':
                try:
                    stats = pipeline.get_statistics()
                    print(f"📊 数据库统计:")
                    print(f"   总文档数: {stats.get('total_documents', 'N/A')}")
                    print(f"   存储大小: {stats.get('storage_size_mb', 'N/A'):.2f}MB")
                    print(f"   向量维度: {stats.get('vector_dimension', 'N/A')}")
                except Exception as e:
                    print(f"❌ 获取统计信息失败: {e}")
                continue
            
            if not query:
                continue
            
            print(f"🔍 搜索中...")
            start_time = time.time()
            
            try:
                search_results = pipeline.search_documents(query, n_results=3)
                
                if search_results:
                    print(f"✅ 找到 {len(search_results)} 个相关结果 (耗时: {time.time() - start_time:.2f}秒):")
                    for i, result in enumerate(search_results, 1):
                        print(f"\n📄 结果 {i}:")
                        print(f"   相似度: {result.get('distance', 'N/A'):.4f}")
                        print(f"   来源: {result.get('metadata', {}).get('source', 'N/A')}")
                        print(f"   内容: {result.get('text', '')}")
                else:
                    print("❌ 未找到相关结果")
                    
            except Exception as e:
                print(f"❌ 搜索失败: {e}")
                
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")


def main():
    """主函数"""
    print("🚀 开始文档问答测试")
    print("=" * 60)
    
    try:
        # 测试基本问答功能
        success = test_document_qa()
        
        if success:
            # 询问是否进行交互式测试
            response = input("\n💬 是否进行交互式问答测试？(y/n): ").strip().lower()
            if response in ['y', 'yes', '是']:
                test_interactive_qa()
        
        print("\n✅ 测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 