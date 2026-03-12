# 🚀 Industrial-Grade RAG Optimization: Structure-Aware Chunking & Hybrid Retrieval
**基于开源框架的工业级 RAG 检索架构重构与评测闭环**

在真实落地大语言模型（LLM）的检索增强生成（RAG）项目时，原生框架往往在处理“结构化文档截断”、“生僻代码/公式召回”以及“多轮意图漂移”时存在严重缺陷。本项目基于 Langchain-Chatchat 底层进行深度二次开发，不仅攻克了 RAG 链路中的四大核心痛点，还独立构建了数据驱动的自动化评估管线，实现了从架构魔改到指标量化的完整 MLOps 闭环。

---

## 💡 核心优化亮点 (Key Contributions)

### 1. 结构感知分块防截断 (Structure-Aware Chunking)
* **痛点 (Issue):** 原生 `RecursiveCharacterTextSplitter` 采用基于固定长度（如 `CHUNK_SIZE=250`）的硬切分，极易导致长代码块（如复杂 Java 算法）和多行 LaTeX 数学公式被强行腰斩，引发大模型严重的上下文幻觉。
* **解决方案 (Solution):** 重写底层 TextSplitter。利用正则表达式在切分前对核心结构（````代码块````、`$$公式$$`）进行**占位符预提取（Placeholder Extraction）**，切分后再进行强制还原。
* **效果 (Result):** 实现了核心代码与公式 **0% 的截断破坏率**，大模型对代码逻辑和状态转移方程的解读准确率大幅跃升。

### 2. 多级漏斗混合检索与融合 (Cascade Hybrid Retrieval & RRF)
* **痛点 (Issue):** 单一的 FAISS 向量检索（Bi-Encoder）对特殊变量名（如自定义的 `maxans`）、生僻报错代码存在“Out of Vocabulary (OOV)”问题，导致精准匹配失效。
* **解决方案 (Solution):** 深入 `knowledge_base_chat.py` 核心调度流，设计并手写了**级联混合检索架构**：
  1. 利用 FAISS 向量域进行高召回率的 Top-20 粗排。
  2. 引入 `jieba` 分词与 `rank_bm25`，在粗排候选池中实施高精度的词频匹配。
  3. 手写**倒数秩融合算法 (Reciprocal Rank Fusion, RRF)**，将语义相似度打分与 BM25 字面匹配打分进行交叉融合。
* **效果 (Result):** 彻底解决了极端 Edge Case 下自创变量名的召回断层问题，实现了“语义+细节”的双维度精准捕捉。

### 3. 交叉重排器的健壮性优化 (Cross-Encoder OOM Prevention)
* **痛点 (Issue):** 引入 `BGE-Reranker-Large` 模型后，由于防截断算法保留了完整的超长代码块，导致传入 Reranker 的 Token 数量超过 XLM-RoBERTa 底层固化的 512 绝对位置编码限制，引发 `Tensor sizes mismatch` 和显存溢出 (OOM)。
* **解决方案 (Solution):** 修改 `reranker.py` 模型加载逻辑，在初始化阶段注入 `max_length=512` 的安全截断锁，在保证头部核心语义不丢失的前提下，完美规避了长序列崩溃问题，保障了重排阶段的高可用性。

### 4. 上下文感知与意图扩展 (Context-Aware HyDE)
* **痛点 (Issue):** 在多轮对话中，用户的追问往往包含代词或极度口语化（如“那这个算法的返回值呢？”），导致纯向量检索因缺失核心主语而发生“检索漂移”。
* **解决方案 (Solution):** 在进入检索漏斗前，设计并植入意图拦截器。将用户的历史对话（Chat History）与当前短 Query 结合，调用 LLM 实时生成假设性文档（Hypothetical Document Embeddings, HyDE），将口语化短句扩写为富含专业术语的“超级查询词”。
* **效果 (Result):** 成功在底层实现了“FAISS 吃扩写长文重语义，BM25 吃原始短文抠字眼”的配合，大幅增强了多轮连问场景下的鲁棒性。

### 5. 数据驱动的 LLM-as-a-Judge 自动化评估管线
* **痛点 (Issue):** RAG 架构的调优极易陷入“过拟合”陷阱，仅靠人工抽查少数 Case 无法量化架构改动对系统整体（幻觉率、相关性）的影响。
* **解决方案 (Solution):** 1. 独立构建了涵盖代码逻辑、特殊变量、跨段落推理的 **Golden Dataset (黄金测试集)**。
  2. 剥离了框架本身的测试局限，手写开发了基于 **LLM-as-a-Judge** 思想的自动化评测脚本 (`auto_evaluator.py`)。
  3. 定义了 Faithfulness (事实忠实度)、Answer Relevance (回答相关性) 和 Correctness (准确率) 三大通用评估维度。
* **效果 (Result):** 实现了 RAG 系统的 MLOps 闭环，能够在一键运行后自动输出多维度的雷达图成绩，为所有的架构魔改提供了坚实的数据支撑。

---

## 📂 核心代码导航 (Project Structure)

为了便于 Code Review，本项目提取了最核心的底层魔改代码（对应原始框架的深层逻辑）：

```text
├── core_algorithms/                  
│   ├── structure_aware_splitter.py   # [核心] 基于正则占位符的结构感知切分器
│   ├── hybrid_rrf_retrieval.py       # [核心] 异步级联混合检索、RRF 融合与 HyDE 意图拦截
│   └── safe_reranker_loader.py       # [修复] 突破 OOM 限制的重排器健壮性加载
│
├── evaluation_pipeline/              
│   ├── evaluation_dataset.jsonl      # [数据] 涵盖四大维度的黄金测试集 (Golden Dataset)
│   └── auto_evaluator.py             # [脚本] 基于原生接口的 LLM-as-a-Judge 自动化裁判打分系统
│
├── configs/                          
│   └── kb_config.py                  # 全局分词器调度配置
│
└── benchmarks/                       

    └── test_chunking.md              # 包含复杂 Java 动态规划与 LaTeX 公式极限测试文档
```

## 🛠️ 部署与快速复现指南 (Deployment & Reproduction)

> **⚠️ 特别声明 (Notice):** > 为了保持核心算法的纯净度并突出本项目的增量贡献 (Incremental Contributions)，本仓库**并未**包含 Langchain-Chatchat 数百兆的原生基础代码。本仓库提供的所有 `.py` 文件均设计为**即插即用的核心补丁 (Core Patches)**。

如果您希望在本地环境中完全复现本项目的双路召回、HyDE 意图拦截以及自动化评测闭环，请按照以下标准流程操作：

### Step 1: 准备基础框架 (Base Framework Setup)
首先，请克隆 Langchain-Chatchat 的官方原生仓库，并完成基础环境与依赖的安装：
```bash
git clone https://github.com/chatchat-space/Langchain-Chatchat.git
cd Langchain-Chatchat
# 按照官方指南完成 pip install -r requirements.txt 等初始化工作
```

### Step 2: 注入核心魔改补丁 (Inject Core Patches)
将本仓库中的核心算法文件，精准覆盖到原生框架的对应深层目录中，完成底层逻辑的“换血”：

1. **防截断分词器拦截：** 将本仓库 `core_algorithms/structure_aware_splitter.py` 覆盖至原生路径 `text_splitter/chinese_recursive_text_splitter.py`。
2. **混合检索与 HyDE 中枢：** 将本仓库 `core_algorithms/hybrid_rrf_retrieval.py` 覆盖至原生路径 `server/chat/knowledge_base_chat.py`。
3. **重排器健壮性保护锁：** 将本仓库 `core_algorithms/safe_reranker_loader.py` 覆盖至原生路径 `server/reranker/reranker.py`。
4. **加载全局配置：**
   将本仓库 `configs/kb_config.py` 覆盖原生目录下的对应配置文件，以激活自定义的分词器。

### Step 3: 一键启动与自动化评测 (Automated Evaluation)
补丁注入完成后，按照常规方式启动 Chatchat 服务：
```bash
python startup.py -a
```
服务成功运行后，**无需人工进行盲测**。请直接将本仓库 `evaluation_pipeline/` 目录下的测试集和裁判脚本拷贝至根目录，一键启动大模型量化打分：
```bash
# 启动 LLM-as-a-Judge 多维度裁判系统
python auto_evaluator.py
```
终端将自动读取 `evaluation_dataset.jsonl` 中的极限 Edge Cases，为您实时输出包含“事实忠实度 (Faithfulness)”、“回答相关性 (Relevance)”与“核心准确率 (Correctness)”的三维雷达图打分报告。

