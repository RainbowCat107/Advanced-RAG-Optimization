from configs import CACHED_VS_NUM, CACHED_MEMO_VS_NUM
from server.knowledge_base.kb_cache.base import *
from server.knowledge_base.kb_service.base import EmbeddingsFunAdapter
from server.utils import load_local_embeddings
from server.knowledge_base.utils import get_vs_path
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
import os


def _new_ds_search(self, search: str) -> Union[str, Document]:
    if search not in self._dict:
        return f"ID {search} not found."
    doc = self._dict[search]
    if isinstance(doc, Document):
        doc.metadata["id"] = search
    return doc


InMemoryDocstore.search = _new_ds_search


class ThreadSafeFaiss(ThreadSafeObject):
    def __repr__(self) -> str:
        cls = type(self).__name__
        docs_count = self.docs_count() if self._obj is not None else 0
        return f"<{cls}: key: {self.key}, obj: {self._obj}, docs_count: {docs_count}>"

    def docs_count(self) -> int:
        return len(self._obj.docstore._dict)

    def save(self, path: str, create_path: bool = True):
        with self.acquire():
            if not os.path.isdir(path) and create_path:
                os.makedirs(path)
            ret = self._obj.save_local(path)
            logger.info(f"Saved vector store {self.key} to disk.")
        return ret

    def clear(self):
        ret = []
        with self.acquire():
            ids = list(self._obj.docstore._dict.keys())
            if ids:
                ret = self._obj.delete(ids)
                assert len(self._obj.docstore._dict) == 0
            logger.info(f"Cleared vector store {self.key}.")
        return ret


class _FaissPool(CachePool):
    def _embedding_dimension(self, embed_model: str) -> int:
        if embed_model in {"text-embedding-v3", "text-embedding-v4"}:
            return int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS") or os.getenv("DASHSCOPE_EMBEDDING_DIMENSIONS") or 1024)
        if embed_model in {"text-embedding-ada-002", "text-embedding-3-small"}:
            return 1536
        if embed_model == "text-embedding-3-large":
            return 3072
        return 0

    def new_vector_store(
        self,
        embed_model: str = EMBEDDING_MODEL,
        embed_device: str = embedding_device(),
    ) -> FAISS:
        embeddings = EmbeddingsFunAdapter(embed_model)
        dim = self._embedding_dimension(embed_model)
        if dim:
            import faiss
            return FAISS(
                embeddings,
                faiss.IndexFlatL2(dim),
                InMemoryDocstore({}),
                {},
                normalize_L2=True,
                distance_strategy="METRIC_INNER_PRODUCT",
            )

        doc = Document(page_content="init", metadata={})
        vector_store = FAISS.from_documents(
            [doc],
            embeddings,
            normalize_L2=True,
            distance_strategy="METRIC_INNER_PRODUCT",
        )
        ids = list(vector_store.docstore._dict.keys())
        vector_store.delete(ids)
        return vector_store

    def save_vector_store(self, kb_name: str, path: str = None):
        if cache := self.get(kb_name):
            return cache.save(path)

    def unload_vector_store(self, kb_name: str):
        if cache := self.get(kb_name):
            self.pop(kb_name)
            logger.info(f"Unloaded vector store: {kb_name}")


class KBFaissPool(_FaissPool):
    def load_vector_store(
            self,
            kb_name: str,
            vector_name: str = None,
            create: bool = True,
            embed_model: str = EMBEDDING_MODEL,
            embed_device: str = embedding_device(),
    ) -> ThreadSafeFaiss:
        self.atomic.acquire()
        vector_name = vector_name or embed_model
        key = (kb_name, vector_name)
        cache = self.get(key)
        if cache is None:
            item = ThreadSafeFaiss(key, pool=self)
            self.set(key, item)
            with item.acquire(msg="init"):
                self.atomic.release()
                try:
                    logger.info(f"loading vector store in '{kb_name}/vector_store/{vector_name}' from disk.")
                    vs_path = get_vs_path(kb_name, vector_name)

                    if os.path.isfile(os.path.join(vs_path, "index.faiss")):
                        embeddings = self.load_kb_embeddings(
                            kb_name=kb_name,
                            embed_device=embed_device,
                            default_embed_model=embed_model,
                        )
                        vector_store = FAISS.load_local(
                            vs_path,
                            embeddings,
                            normalize_L2=True,
                            distance_strategy="METRIC_INNER_PRODUCT",
                        )
                    elif create:
                        if not os.path.exists(vs_path):
                            os.makedirs(vs_path)
                        vector_store = self.new_vector_store(
                            embed_model=embed_model,
                            embed_device=embed_device,
                        )
                        vector_store.save_local(vs_path)
                    else:
                        raise RuntimeError(f"knowledge base {kb_name} not exist.")

                    item.obj = vector_store
                    item.finish_loading()
                except Exception:
                    self.pop(key)
                    item.finish_loading()
                    raise
        else:
            self.atomic.release()
        return self.get(key)


class MemoFaissPool(_FaissPool):
    def load_vector_store(
        self,
        kb_name: str,
        embed_model: str = EMBEDDING_MODEL,
        embed_device: str = embedding_device(),
    ) -> ThreadSafeFaiss:
        self.atomic.acquire()
        cache = self.get(kb_name)
        if cache is None:
            item = ThreadSafeFaiss(kb_name, pool=self)
            self.set(kb_name, item)
            with item.acquire(msg="init"):
                self.atomic.release()
                try:
                    logger.info(f"loading vector store in '{kb_name}' to memory.")
                    vector_store = self.new_vector_store(
                        embed_model=embed_model,
                        embed_device=embed_device,
                    )
                    item.obj = vector_store
                    item.finish_loading()
                except Exception:
                    self.pop(kb_name)
                    item.finish_loading()
                    raise
        else:
            self.atomic.release()
        return self.get(kb_name)


kb_faiss_pool = KBFaissPool(cache_num=CACHED_VS_NUM)
memo_faiss_pool = MemoFaissPool(cache_num=CACHED_MEMO_VS_NUM)
