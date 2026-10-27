from langchain_chroma import Chroma
from chromadb.config import Settings
import config_data as config

class VectorStoreService(object):
    def __init__(self,embedding):
        """
        :param embedding: 嵌入模型的传入
        """
        self.embedding= embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
            # Chroma 0.4.x 与新版 posthog 的匿名遥测接口不兼容；
            # 本地毕设不需要遥测，关闭它可避免无关的 capture 警告。
            client_settings=Settings(anonymized_telemetry=False),
        )

    def get_retriever(self):
            """返回向量检索器，方便加入chain"""
            return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})

if __name__ =='__main__':
        from langchain_community.embeddings import DashScopeEmbeddings
        retriever= VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()

        res= retriever.invoke("我的身高180，尺码推荐")
        print(res)
