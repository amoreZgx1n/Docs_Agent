import asyncio
import pandas as pd
import numpy as np
import torch
from typing import Dict, Union
from dataclasses import dataclass
from datetime import datetime
from llama_index.llms.ollama import Ollama
import sentence_transformers
from sentence_transformers import SentenceTransformer, util

@dataclass
class AnalysisResult:
    """数据分析结果"""
    data: pd.DataFrame
    query: str

@dataclass
class DocQnAResult:
    """文档问答结果"""
    context: str
    metadata: Dict
    query: str

# 意图识别
class StrictIntentProcessor:
    def __init__(self, embedding_model: SentenceTransformer):
        self.llm = Ollama(model="llama3.1", request_timeout=30.0)
        self.embedding_model = embedding_model

    async def process(
        self,
        query: str,
        data: pd.DataFrame
    ) -> Union[AnalysisResult, DocQnAResult]:
        if 'vector' in data.columns and 'text_chunk' in data.columns:
            return self._find_best_match(query, data)
        else:
            return AnalysisResult(data=data, query=query)

    def _find_best_match(self, query: str, data: pd.DataFrame) -> DocQnAResult:
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
            query=query
        )

# 辅助函数：成一个经过向量化的知识库DataFrame
def create_vectorized_kb(model: SentenceTransformer) -> pd.DataFrame:

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
    print("正在加载文本编码模型...")
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("模型加载完毕。")
    
    processor = StrictIntentProcessor(embedding_model=embedding_model)

    print("\n\n----------- 1. 文档问答任务测试 -----------")
    vector_kb_df = create_vectorized_kb(embedding_model)
    print("\n[输入知识库]")
    print(vector_kb_df[['text_chunk', 'source']])
    print("-------------------------------------------")

    query_doc = "AI的关键是什么？"
    doc_qa_result = await processor.process(query_doc, vector_kb_df)
    
    print("\n[输出结果]")
    print(f"处理结果类型: {type(doc_qa_result)}")
    if isinstance(doc_qa_result, DocQnAResult):
        print(f"原始问题: {doc_qa_result.query}")
        print(f"匹配到的上下文: {doc_qa_result.context}")
        print(f"详细元数据: {doc_qa_result.metadata}")

    print("\n\n----------- 2. 数据分析任务测试 -----------")
    
    df_sales = pd.DataFrame({
        "产品ID": ["A101", "A102", "B201"],
        "日期": pd.to_datetime(["2025-08-18", "2025-08-18", "2025-08-19"]),
        "销售额": [150.0, 200.5, 75.0]
    })
    
    print("\n[输入数据]")
    print(df_sales)
    print("-------------------------------------------")
    
    query_analysis = "8月18日哪款产品销量最高？"
    analysis_result = await processor.process(query_analysis, df_sales)
    
    print(f"\n[输出结果]")
    print(f"处理结果类型: {type(analysis_result)}")
    if isinstance(analysis_result, AnalysisResult):
        print(f"原始问题: {analysis_result.query}")
        print(f"数据：{analysis_result.data}")

        
if __name__ == "__main__":
    asyncio.run(main())