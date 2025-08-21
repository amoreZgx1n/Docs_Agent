# Docs_Agent

## 2025腾讯犀牛鸟开源人才培养计划—腾讯文档项目

***

## 🚀 项目概述

### 这是一个基于大语言模型的智能文档问答和数据分析Agent系统，支持PDF文档处理和表格数据分析。

***

## 📋 功能特性

### 📄 文档处理模块（数据摄入与预处理）
- **PDF文档处理**: 支持PDF转Markdown，文档清洗和分块
- **表格数据处理**: 支持CSV/XLSX文件读取和DataFrame转换
- **文本分块**: 智能文本切分，支持多种分块策略
- **向量化**: 使用SiliconFlow API进行文本向量化
- **向量存储**: 基于ChromaDB的向量数据库存储

### 🧠 意图识别模块（查询理解与路由）
- **意图识别**: 智能识别用户查询意图
- **文档问答**: 基于向量检索的文档问答功能
- **数据分析**: 表格数据的智能分析
- **混合检索**: BM25和密集检索的混合搜索

### 🤖 LLM处理模块（智能生成与输出）
- **代码生成**: 自动生成Python代码
- **代码执行**: 安全的Python沙箱执行环境
- **图表生成**: 自动生成数据可视化图表
- **报告生成**: 智能生成分析报告   

***

## 🚩完成进度：
- [ ] 文档处理、清洗、向量化、向量存储、检索
- [ ] 数据分析 or 文档问答 路由
- [ ] LLM代码生成，数据探索，数据分析
- [ ] 单轮对话
- [ ] 意图识别
- [ ] 长短期记忆管理，多轮对话
- [ ] 长表格处理、分析, SQL生成
- [ ] 分析图表生成，数据可视化

***

## 🛠️ 安装与配置

### 1. 克隆项目
```bash
git clone https://github.com/amoreZgx1n/Docs_Agent.git
cd Docs_Agent
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置环境变量
创建 `.env` 文件：
```env
SILICONFLOW_API_KEY=your_api_key_here
ZHIPU_API_KEY =
ZHIPU_BASE_URL =
ZHIPU_MODEL =
```

### 4. 配置Ollama（可选）
如果需要本地LLM支持，请安装Ollama：
```bash
# 安装Ollama
# 下载模型
ollama pull qwen3:8b
```

***

## 📖 使用方法

### 基本使用
```python
import asyncio
from selector import EnhancedIntentProcessor
import sentence_transformers

async def main():
    # 初始化处理器
    embedding_model = sentence_transformers.SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    processor = EnhancedIntentProcessor(embedding_model=embedding_model)
    
    # 处理PDF文档
    result = await processor.process_query("文档内容是什么？", file_path="path/to/document.pdf")
    
    # 处理表格数据
    result = await processor.process_query("销售额最高的产品是什么？", file_path="path/to/sales.csv")
    
    # 搜索文档
    results = await processor.search_documents("关键词", top_k=5)

asyncio.run(main())
```

### 运行测试
```bash
# 运行集成测试
python test_integration.py

# 运行主程序
python selector.py
```

***

## 📁 项目结构

```
Docs_Agent/
├── src/                    # 核心源代码
│   ├── document_processor/ # 文档处理模块
│   ├── text_processor/     # 文本处理模块
│   ├── storage/           # 存储模块
│   ├── llm_generate/      # LLM处理模块
│   ├── pipeline/          # 主流水线
│   └── utils/             # 工具模块
├── data/                  # 数据目录
│   └── documents/         # 文档存储
├── tools/                 # 工具模块
├── logs/                  # 日志文件
├── selector.py            # 主入口文件
├── config.yaml            # 配置文件
├── requirements.txt       # 依赖列表
└── README.md             # 项目说明
```

***

## 🔧 配置说明

主要配置在 `config.yaml` 文件中：

```yaml
document_processing:
  chunk_size: 300
  chunk_overlap: 30
  
embedding:
  api:
    url: "https://api.siliconflow.cn/v1/embeddings"
    model: "netease-youdao/bce-embedding-base_v1"
    max_tokens: 512
    
storage:
  vector_db:
    persist_directory: "./chroma_db"
```

***

## 🤝 团队分工

- **文档处理模块**: 数据摄入与预处理（PDF处理、表格处理、向量化）
- **意图识别模块**: 查询理解与路由（意图识别、文档问答、数据分析）
- **LLM处理模块**: 智能生成与输出（代码生成、执行、图表生成）

***

## 📝 开发说明

### Git提交步骤
1. 拉取最新代码: `git pull origin main`
2. 开发完成后: `git status`
3. 添加所有修改: `git add .`
4. 提交: `git commit -m "评论"`
5. 再次拉取: `git pull origin main`
6. 推送: `git push origin main`

***

## 📄 许可证

本项目为2025腾讯犀牛鸟开源人才培养计划项目。

***

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

**最后更新**: 2025-08-20
**版本**: v1.0.0
