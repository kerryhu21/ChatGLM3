import os

from langchain.embeddings import HuggingFaceEmbeddings

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
EMBEDDING_PATH = os.environ.get('EMBEDDING_PATH', 'BAAI/bge-m3')


if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_PATH)
    query_result = embeddings.embed_query("Hello world")
    doc_result = embeddings.embed_documents(["Hello world", "Bye world"])
    print(f"query_result {query_result} \n\n")
    print(f"doc_result {doc_result}")