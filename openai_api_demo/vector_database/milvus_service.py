from pymilvus import MilvusClient
from pymilvus import model


COLLECTION_NAME = "demo_collection"
DATABASE_NAME = "market.db"


def init():
    global client
    global embedding_fn
    client = MilvusClient(DATABASE_NAME)
    embedding_fn = model.DefaultEmbeddingFunction()


def create_collection(collection_name):
    if client.has_collection(collection_name=collection_name):
        return
    # create collection
    client.create_collection(
        collection_name=collection_name,
        dimension=768,  # The vectors we will use in this demo has 768 dimensions
    )


def embedding_docs(docs):
    vectors = embedding_fn.encode_documents(docs)
    print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)
    # build data structure: id | vector | text | subject
    data = [
        {"id": i, "vector": vectors[i], "text": docs[i], "scene": "产品营销"}
        for i in range(len(vectors))
    ]

    print("Data has", len(data), "entities, each with fields: ", data[0].keys())
    print("Vector dim:", len(data[0]["vector"]))
    return data


def insert_data(collection_name, data):
    create_collection(collection_name)
    docs = embedding_docs(data)
    client.insert(collection_name=collection_name, data=docs)


def query_content_by_vector(query):
    query_vectors = embedding_fn.encode_documents([query])
    result = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vectors,
        output_fields=["text", "subject"],
        limit=1,
    )
    print("query result: ", result)


def query_content_by_vector_scalar(query, filter):
    result = client.search(
        collection_name=COLLECTION_NAME,
        data=embedding_fn.encode_queries([query]),
        filter=filter,
        limit=2,
        output_fields=["text", "subject"],
    )
    print("query vector and scalar result: ", result)


if __name__ == "__main__":
    init()
    # create_vector_database()
    # data = embedding_docs()
    # insert_data(data)
    # query_content_by_vector("Who is Alan Turing?")
    query_content_by_vector_scalar(
        "tell me AI related information", "subject == 'history'"
    )
