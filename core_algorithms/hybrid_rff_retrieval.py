from fastapi import Body, Request
from sse_starlette.sse import EventSourceResponse
from fastapi.concurrency import run_in_threadpool
from configs import (LLM_MODELS, 
                     VECTOR_SEARCH_TOP_K, 
                     SCORE_THRESHOLD, 
                     TEMPERATURE,
                     USE_RERANKER,
                     RERANKER_MODEL,
                     RERANKER_MAX_LENGTH,
                     MODEL_PATH)
from server.utils import wrap_done, get_ChatOpenAI
from server.utils import BaseResponse, get_prompt_template
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBServiceFactory
import json
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs
from server.reranker.reranker import LangchainReranker
from server.utils import embedding_device
import jieba
from rank_bm25 import BM25Okapi
from typing import List
def reciprocal_rank_fusion(vector_docs: List, bm25_docs: List, k: int = 60) -> List:
    """
    【手撕 RRF 融合算法】
    公式: Score = 1 / (k + rank)
    将两路召回的结果根据排名进行交叉计分融合。
    """
    rrf_scores = {}
    doc_map = {} # 用于保存文档对象

    # 1. 遍历向量检索结果打分
    for rank, doc in enumerate(vector_docs):
        # 使用文档内容作为唯一标识符 ID
        doc_id = doc.page_content 
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    # 2. 遍历 BM25 关键词检索结果打分
    for rank, doc in enumerate(bm25_docs):
        doc_id = doc.page_content
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    # 3. 根据 RRF 总分对文档进行降序重新排列
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # 4. 返回融合后的纯文档列表
    fused_docs = [doc_map[doc_id] for doc_id, score in sorted_docs]
    return fused_docs
async def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                              knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                              top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                              score_threshold: float = Body(
                                  SCORE_THRESHOLD,
                                  description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                  ge=0,
                                  le=2
                              ),
                              history: List[History] = Body(
                                  [],
                                  description="历史对话",
                                  examples=[[
                                      {"role": "user",
                                       "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                      {"role": "assistant",
                                       "content": "虎头虎脑"}]]
                              ),
                              stream: bool = Body(False, description="流式输出"),
                              model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                              temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                              max_tokens: Optional[int] = Body(
                                  None,
                                  description="限制LLM生成Token数量，默认None代表模型最大值"
                              ),
                              prompt_name: str = Body(
                                  "default",
                                  description="使用的prompt模板名称(在configs/prompt_config.py中配置)"
                              ),
                              request: Request = None,
                              ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History.from_data(h) for h in history]

    async def knowledge_base_chat_iterator(
            query: str,
            top_k: int,
            history: Optional[List[History]],
            model_name: str = model_name,
            prompt_name: str = prompt_name,
    ) -> AsyncIterable[str]:
        nonlocal max_tokens
        callback = AsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )
        # === [原版单路召回（已注释）] ===
        # docs = await run_in_threadpool(search_docs,
        #                                query=query,
        #                                knowledge_base_name=knowledge_base_name,
        #                                top_k=top_k,
        #                                score_threshold=score_threshold)

        # === [你的硬核魔改：BM25 + FAISS 混合检索与 RRF 融合] ===
        # 第一阶段：先让 FAISS 向量库放宽视野，捞取 Top-20 的粗排候选池
        vector_docs = await run_in_threadpool(search_docs,
                                              query=query,
                                              knowledge_base_name=knowledge_base_name,
                                              top_k=20, # 放大召回池
                                              score_threshold=score_threshold)

        # 第二阶段：在候选池中实施 BM25 精确关键词匹配
        if vector_docs:
            # 1. 将文档分词构建 BM25 语料库
            tokenized_corpus = [list(jieba.cut(doc.page_content)) for doc in vector_docs]
            bm25 = BM25Okapi(tokenized_corpus)
            # 2. 将用户的提问分词
            tokenized_query = list(jieba.cut(query))
            # 3. 算出候选文档的字面匹配分数，并排序
            bm25_scores = bm25.get_scores(tokenized_query)
            bm25_scored_docs = sorted(zip(vector_docs, bm25_scores), key=lambda x: x[1], reverse=True)
            # 4. 剔除掉完全没有关键词的文档（分数为0）
            bm25_docs = [doc for doc, score in bm25_scored_docs if score > 0]
        else:
            bm25_docs = []

        # 第三阶段：调用你手撕的 RRF 算法，将“语义榜单”和“字眼榜单”完美融合！
        fused_docs = reciprocal_rank_fusion(vector_docs, bm25_docs)

        # 第四阶段：只截取最精华的 Top-K 篇文档交给下游（Reranker）
        docs = fused_docs[:top_k]
        # =======================================================

        # 加入reranker
        if USE_RERANKER:
            reranker_model_path = MODEL_PATH["reranker"].get(RERANKER_MODEL,"BAAI/bge-reranker-large")
            print("-----------------model path------------------")
            print(reranker_model_path)
            reranker_model = LangchainReranker(top_n=top_k,
                                            device=embedding_device(),
                                            max_length=RERANKER_MAX_LENGTH,
                                            model_name_or_path=reranker_model_path
                                            )
            print(docs)
            docs = reranker_model.compress_documents(documents=docs,
                                                     query=query)
            print("---------after rerank------------------")
            print(docs)
        context = "\n".join([doc.page_content for doc in docs])

        if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
            prompt_template = get_prompt_template("knowledge_base_chat", "empty")
        else:
            prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = []
        for inum, doc in enumerate(docs):
            filename = doc.metadata.get("source")
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
            base_url = request.base_url
            url = f"{base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if len(source_documents) == 0:  # 没有找到相关文档
            source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)
        await task

    return EventSourceResponse(knowledge_base_chat_iterator(query, top_k, history,model_name,prompt_name))

