import asyncio
import pandas as pd
import numpy as np
import torch
import os
import sys
from typing import Dict, Union, List, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from llama_index.llms.ollama import Ollama
import sentence_transformers
from sentence_transformers import SentenceTransformer, util

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入你的文档处理模块
from src.utils.logger import setup_logger
from src.pipeline.document_pipeline import DocumentPipeline
from src.utils.config import get_config

from src.llm_generate.data_analyze2 import analyze_data
from src.llm_generate.doc_qa import doc_qa

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['SILICONFLOW_API_KEY'] = 'sk-ealsjxzyweovkgxqlshbuophkshxjwaiawrldqnvlhzhsatd'
@dataclass
class AnalysisResult:
    """数据分析结果"""
    data: pd.DataFrame
    query: str
    source: str

@dataclass
class DocQnAResult:
    """文档问答结果"""
    context: str
    metadata: Dict
    query: str
    source: str

@dataclass
class ProcessingResult:
    """文档处理结果"""
    success: bool
    message: str
    data: Optional[Union[pd.DataFrame, List[Dict]]] = None

class EnhancedIntentProcessor:
    """增强的意图处理器，集成文档处理与意图识别功能"""
    
    def __init__(self, embedding_model: SentenceTransformer):
        self.llm = Ollama(model="qwen3:8b", request_timeout=30.0)
        self.embedding_model = embedding_model
        
        # 初始化日志和配置
        setup_logger()
        self.config = get_config()
        
        # 初始化文档处理流水线
        self.document_pipeline = DocumentPipeline()
        
        print("✅ 增强意图处理器初始化完成")

    async def process_query(
        self,
        query: str,
        data: Optional[pd.DataFrame] = None,
        file_path: Optional[str] = None
    ) -> Union[AnalysisResult, DocQnAResult, ProcessingResult]:
        """
        处理用户查询
        
        Args:
            query: 用户查询
            data: 已有的数据（可选）
            file_path: 文件路径（可选）
        
        Returns:
            处理结果
        """
        
        # 如果有文件路径，先处理文件
        if file_path:
            processing_result = await self._process_document(file_path)
            if not processing_result.success:
                return processing_result
            
            # 如果处理的是表格文件，返回AnalysisResult
            if file_path.lower().endswith(('.csv', '.xlsx', '.xls')):
                return AnalysisResult(
                    data=processing_result.data,
                    query=query,
                    source=file_path
                )
        
        # 如果有向量化数据，进行文档问答
        if data is not None and 'vector' in data.columns and 'text_chunk' in data.columns:
            best = self._find_best_match(query, data)
            def to_snippets(x: Union[DocQnAResult, List[DocQnAResult]]) -> List[Dict[str, str]]:
                if isinstance(x, list):
                    return [{"source": i.source, "text": i.context} for i in x]
                else:
                    return [{"source": x.source, "text": x.context}]
            snippets = to_snippets(best)
            
            doc_qa_result = doc_qa(query, snippets)
            return doc_qa_result
        
        # 如果有表格数据，进行数据分析
        elif data is not None:
            # return AnalysisResult(
            #     data=data,
            #     query=query,
            #     source="provided_data"
            # )
            return analyze_data(query, data)
        
        # 默认返回处理结果
        return ProcessingResult(
            success=True,
            message="查询已接收，请提供数据或文件进行处理",
            data=None
        )

    async def _process_document(self, file_path: str) -> ProcessingResult:
        """处理文档文件"""
        try:
            print(f"📄 开始处理文档: {file_path}")
            
            # 使用文档处理流水线
            result = self.document_pipeline.process_document(file_path)
            
            if result['success']:
                print(f"✅ 文档处理成功: {file_path}")
                return ProcessingResult(
                    success=True,
                    message=f"文档 {Path(file_path).name} 处理成功",
                    data=result.get('data')
                )
            else:
                print(f"❌ 文档处理失败: {file_path}")
                return ProcessingResult(
                    success=False,
                    message=f"文档处理失败: {result.get('error', '未知错误')}"
                )
                
        except Exception as e:
            print(f"❌ 文档处理异常: {str(e)}")
            return ProcessingResult(
                success=False,
                message=f"文档处理异常: {str(e)}"
            )

    def _find_best_match(self, query: str, data: pd.DataFrame) -> DocQnAResult:
        """在向量化数据中查找最佳匹配"""
        try:
            query_vector = self.embedding_model.encode(query, convert_to_tensor=True)
            corpus_vectors_np = np.stack(data['vector'].values)
            corpus_vectors = torch.from_numpy(corpus_vectors_np).to(query_vector.device)
            cosine_scores = util.cos_sim(query_vector, corpus_vectors)[0]
            best_match_idx = np.argmax(cosine_scores.cpu().numpy())
            best_match_score = cosine_scores[best_match_idx]
            best_match_row = data.iloc[best_match_idx]

            matched_context = best_match_row['text_chunk']
            
            metadata = {
                "source": best_match_row.get('source', 'Unknown'),
                "chunk_index": int(best_match_row.name),
                "similarity_score": float(best_match_score),
                "full_content": matched_context,
                "processed_at": datetime.now().isoformat(),
            }

            return DocQnAResult(
                context=matched_context,
                metadata=metadata,
                query=query,
                source=best_match_row.get('source', 'Unknown')
            )
        except Exception as e:
            print(f"❌ 向量匹配失败: {str(e)}")
            return DocQnAResult(
                context="匹配失败",
                metadata={"error": str(e)},
                query=query,
                source="error"
            )

    async def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索已处理的文档"""
        try:
            results = self.document_pipeline.search_documents(query, top_k)
            return results
        except Exception as e:
            print(f"❌ 文档搜索失败: {str(e)}")
            return []

    def get_statistics(self) -> Dict:
        """获取处理统计信息"""
        try:
            return self.document_pipeline.get_statistics()
        except Exception as e:
            print(f"❌ 获取统计信息失败: {str(e)}")
            return {"error": str(e)}

# 辅助函数：创建一个经过向量化的知识库DataFrame
def create_vectorized_kb(model: SentenceTransformer) -> pd.DataFrame:
    """创建示例向量化知识库"""
    chunks = [
        "机器学习是人工智能的核心技术。",
        "深度学习在图像和语音识别上表现优异。",
        "可持续发展需要平衡经济、社会和环境三者。",
        "企业社会责任是公司对社会贡献的体现。",
    ]
    
    print("正在使用模型对文本进行编码，请稍候...")
    vectors = model.encode(chunks)
    print("编码完成。")
    
    df = pd.DataFrame({
        "text_chunk": chunks,
        "vector": list(vectors),
        "source": ["doc_ai.txt", "doc_ai.txt", "doc_sustainability.txt", "doc_sustainability.txt"]
    })
    
    return df

async def main():
    """主函数 - 测试集成后的功能"""
    print("🚀 启动增强版文档问答和数据分析系统")
    print("=" * 60)
    
    print("正在加载文本编码模型...")
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("模型加载完毕。")
    
    processor = EnhancedIntentProcessor(embedding_model=embedding_model)
    vector_kb_df = create_vectorized_kb(embedding_model)

    print("\n\n----------- 1. 文档问答任务测试 -----------")
    query_doc = "AI的关键是什么？"
    
    doc_qa_result = await processor.process_query(query_doc, vector_kb_df)
    # doc_qa_result = await processor.process_query(query=query_doc, file_path='./data/documents/深度学习（花书）.pdf')
    
    print("\n[输出结果]")
    print(doc_qa_result)
    # print(f"处理结果类型: {type(doc_qa_result)}")
    # if isinstance(doc_qa_result, DocQnAResult):
    #     print(f"原始问题: {doc_qa_result.query}")
    #     print(f"匹配到的上下文: {doc_qa_result.context}")
    #     print(f"来源: {doc_qa_result.source}")
    #     print(f"详细元数据: {doc_qa_result.metadata}")

    print("\n\n----------- 2. 数据分析任务测试 -----------")
    
    df_sales = pd.read_csv("./data/tables/order_details.csv")
    print("\n[输入数据示例]")
    print(df_sales.head(10))
    print("-------------------------------------------")
    query_analysis = "不同种类商品的售卖数量、售卖总额、利润以及利润占比？"
    analysis_result = await processor.process_query(query_analysis, df_sales)
    print(f"\n[输出结果]")
    print(analysis_result)

    print("\n\n----------- 3. 文档处理功能测试 -----------")
    print("💡 提示：你可以使用以下命令测试文档处理功能：")
    print("   - 处理PDF文件: await processor.process_query('查询内容', file_path='path/to/file.pdf')")
    print("   - 处理表格文件: await processor.process_query('查询内容', file_path='path/to/file.csv')")
    print("   - 搜索文档: await processor.search_documents('搜索关键词')")
    print("   - 获取统计: processor.get_statistics()")

if __name__ == "__main__":
    asyncio.run(main())