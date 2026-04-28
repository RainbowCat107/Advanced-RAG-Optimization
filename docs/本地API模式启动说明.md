# 本地 API 模式启动说明

本项目根目录是 `C:\Users\iDyt107\Desktop\Langchain-Chatchat`。请在这个目录运行 `startup.py`，不要进入内层同名目录。

本项目已配置为默认使用 OpenAI-compatible 大模型 API，不再默认加载 AutoDL 上的本地大模型路径。

## PowerShell 环境变量

```powershell
$env:OPENAI_API_KEY="你的 API Key"
$env:OPENAI_API_BASE="https://api.openai.com/v1"
$env:OPENAI_API_MODEL="gpt-4o-mini"
```

如果你使用的是 One API、New API、硅基流动、阿里百炼等兼容 OpenAI 协议的网关，把 `OPENAI_API_BASE` 和 `OPENAI_API_MODEL` 改成对应服务提供的地址和模型名即可。

阿里云百炼示例：

```powershell
$env:OPENAI_API_KEY="你的阿里云百炼 API Key"
$env:OPENAI_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:OPENAI_API_MODEL="qwen-plus"
$env:LLM_MODELS="openai-api"
$env:EMBEDDING_MODEL="text-embedding-v4"
$env:USE_RERANKER="false"
$env:OPENAI_PROXY=""
```

## Embedding

默认 Embedding 已改为 API 模式，也会读取同一个 `OPENAI_API_KEY` 和 `OPENAI_API_BASE`。

可选覆盖：

```powershell
$env:EMBEDDING_MODEL="text-embedding-3-small"
```

阿里云百炼推荐使用：

```powershell
$env:EMBEDDING_MODEL="text-embedding-v4"
```

注意：如果原来的向量库是用 `bge-large-zh-v1.5` 建的，切换到 OpenAI Embedding 后需要重建向量库，否则 FAISS 向量维度不一致。

```powershell
python init_database.py --recreate-vs
```

## 启动

```powershell
python startup.py -a -i
```

`-i/--lite` 会避免启动本地 model worker，更适合本地 API 模式。
