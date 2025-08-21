"""
简单的文档处理流水线测试脚本
"""

import os
import sys
from pathlib import Path

# 设置环境变量
os.environ['SILICONFLOW_API_KEY'] = 'sk-lpylryqgkbbgxnxlarlrenupxzkfrqlrijyulgdamsegpugs'

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_document_pipeline():
    """测试文档处理流水线"""
    print("🧪 开始文档处理流水线测试")
    print("=" * 50)
    
    try:
        # 导入文档处理流水线
        from src.pipeline.document_pipeline import DocumentPipeline
        print("✅ 文档处理流水线导入成功")
        
        # 初始化流水线
        pipeline = DocumentPipeline()
        print("✅ 文档处理流水线初始化成功")
        
        # 检查是否有测试PDF文件
        test_pdf_path = Path("data/documents/毛选.pdf")
        if not test_pdf_path.exists():
            print(f"⚠️  测试PDF文件不存在: {test_pdf_path}")
            return
        
        print(f"📄 找到测试PDF文件: {test_pdf_path}")
        
        # 测试文档处理
        print("\n🔄 开始文档处理测试...")
        result = pipeline.process_document(test_pdf_path)
        
        print(f"✅ 文档处理成功！")
        print(f"📊 处理结果: {result}")
        
        # 测试搜索功能
        print("\n🔍 测试搜索功能...")
        search_results = pipeline.search_documents("毛泽东", n_results=3)
        print(f"✅ 搜索成功！找到 {len(search_results)} 个结果")
        
        # 测试统计功能
        print("\n📈 测试统计功能...")
        stats = pipeline.get_statistics()
        print(f"✅ 统计成功！数据库统计: {stats}")
        
    except Exception as e:
        print(f"❌ 流水线测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

def test_text_chunker():
    """测试文本分块器"""
    print("\n🧪 开始文本分块器测试")
    print("=" * 50)
    
    try:
        # 导入文本分块器
        from src.text_processor.text_chunker import TextChunker
        print("✅ 文本分块器导入成功")
        
        # 初始化分块器
        chunker = TextChunker()
        print("✅ 文本分块器初始化成功")
        
        # 测试文本分块
        test_text = """
# 测试文档

这是一个测试文档，用于验证文本分块功能。

## 第一部分

这部分包含一些测试内容。

## 第二部分

这部分也包含一些测试内容。

### 子部分

这是子部分的内容。
        """
        
        print("\n🔄 开始文本分块测试...")
        chunks = chunker.chunk_text(test_text, method='basic')
        
        print(f"✅ 文本分块成功！")
        print(f"📊 生成了 {len(chunks)} 个文本块")
        
        for i, chunk in enumerate(chunks[:3]):  # 只显示前3个
            print(f"📝 块 {i+1}: {chunk[:100]}...")
        
    except Exception as e:
        print(f"❌ 文本分块器测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

def test_text_embedder():
    """测试文本向量化器"""
    print("\n🧪 开始文本向量化器测试")
    print("=" * 50)
    
    try:
        # 导入文本向量化器
        from src.text_processor.text_embedder import TextEmbedder
        print("✅ 文本向量化器导入成功")
        
        # 初始化向量化器
        embedder = TextEmbedder()
        print("✅ 文本向量化器初始化成功")
        
        # 测试文本向量化
        test_texts = ["这是一个测试文本", "这是另一个测试文本"]
        
        print("\n🔄 开始文本向量化测试...")
        embeddings = embedder.embed_texts(test_texts)
        
        print(f"✅ 文本向量化成功！")
        print(f"📊 生成了 {len(embeddings)} 个向量")
        print(f"📏 向量维度: {len(embeddings[0]) if embeddings else 0}")
        
    except Exception as e:
        print(f"❌ 文本向量化器测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_text_chunker()
    test_text_embedder()
    test_document_pipeline() 