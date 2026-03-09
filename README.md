# 🚀 Industrial-Grade RAG Optimization: Structure-Aware Chunking & Hybrid Retrieval
**基于开源框架的工业级 RAG 检索架构重构与底层调优**

在真实落地大语言模型（LLM）的检索增强生成（RAG）项目时，原生框架往往在处理“结构化文档截断”和“生僻代码/公式召回”时存在严重缺陷。本项目基于 Langchain-Chatchat 底层进行深度二次开发，重点攻克了 RAG 链路中的三大核心痛点。

---

## 💡 核心优化亮点 (Key Contributions)

### 1. 结构感知分块防截断 (Structure-Aware Chunking)
 **痛点 (Issue):** 原生 `RecursiveCharacterTextSplitter` 采用基于固定长度（如 `CHUNK_SIZE=250`）的硬切分，极易导致长代码块（如复杂 Java 算法）和多行 LaTeX 数学公式被强行腰斩，引发大模型严重的上下文幻觉。
 **解决方案 (Solution):** 重写底层 TextSplitter。利用正则表达式在切分前对核心结构（````代码块````、`$$公式$$`）进行**占位符预提取（Placeholder Extraction）**，切分后再进行强制还原。
 **效果 (Result):** 实现了核心代码与公式 **0% 的截断破坏率**，大模型对代码逻辑和状态转移方程的解读准确率大幅跃升。

### 2. 多级漏斗混合检索与融合 (Cascade Hybrid Retrieval & RRF)
 **痛点 (Issue):** 单一的 FAISS 向量检索（Bi-Encoder）对特殊变量名（如自定义的 `maxans`）、生僻报错代码存在“Out of Vocabulary (OOV)”问题，导致精准匹配失效。
 **解决方案 (Solution):** 深入 `knowledge_base_chat.py` 核心调度流，设计并手写了**级联混合检索架构**：
  1. 利用 FAISS 向量域进行高召回率的 Top-20 粗排。
  2. 引入 `jieba` 分词与 `rank_bm25`，在粗排候选池中实施高精度的词频匹配。
  3. 手写**倒数秩融合算法 (Reciprocal Rank Fusion, RRF)**，将语义相似度打分与 BM25 字面匹配打分进行交叉融合。
 **效果 (Result):** 彻底解决了极端 Edge Case 下自创变量名的召回断层问题，实现了“语义+细节”的双维度精准捕捉。

### 3. 交叉重排器的健壮性优化 (Cross-Encoder OOM Prevention)
 **痛点 (Issue):** 引入 `BGE-Reranker-Large` 模型后，由于防截断算法保留了完整的超长代码块，导致传入 Reranker 的 Token 数量超过 XLM-RoBERTa 底层固化的 512 绝对位置编码限制，引发 `Tensor sizes mismatch` 和显存溢出 (OOM)。
 **解决方案 (Solution):** 修改 `reranker.py` 模型加载逻辑，在初始化阶段注入 `max_length=512` 的安全截断锁，在保证头部核心语义不丢失的前提下，完美规避了长序列崩溃问题，保障了重排阶段的高可用性。

---

## 📂 核心代码导航 (Project Structure)

为了便于 Code Review，本项目提取了最核心的底层魔改代码（对应原始框架的深层逻辑）：

```text
├── core_algorithms/                  
│   ├── structure_aware_splitter.py   #  基于正则占位符的结构感知切分器
│   ├── hybrid_rrf_retrieval.py       #  异步级联混合检索与 RRF 融合算法
│   └── safe_reranker_loader.py       #  突破 OOM 限制的重排器健壮性加载
├── configs/                          
│   └── kb_config.py                  # 全局分词器调度配置
└── benchmarks/                       

    └── test_chunking.md              # 包含复杂 Java 动态规划与 LaTeX 公式极限测试文档
