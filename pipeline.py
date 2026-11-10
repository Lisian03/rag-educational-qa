from ingestion import load_documents, split_documents
from vectorstore import build_vectorstore
from retrieval import retrieve
from reranker import rerank
from generation import Generator
print("LOADING DATA")
print("CREATING EMBEDDINGS")
EMBEDDINGS = {
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "bge": "BAAI/bge-m3"
}

LLMS = {
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct"
}
print("RUNNING RETRIEVAL")
class RAGSystem:

    def __init__(self):
        docs = load_documents()
        chunks = split_documents(docs)

        self.vectorstores = {
            name: build_vectorstore(chunks, model)
            for name, model in EMBEDDINGS.items()
        }

        self.generators = {
            name: Generator(model)
            for name, model in LLMS.items()
        }

    def query(self, question, emb_name, llm_name, use_reranker=True):

        db = self.vectorstores[emb_name]
        docs = retrieve(question, db)

        if use_reranker:
            docs = rerank(question, docs)

        answer = self.generators[llm_name].generate(question, docs)

        sources = list(set([d.metadata["source"] for d in docs]))

        return answer, sources
