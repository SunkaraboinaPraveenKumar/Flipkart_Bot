from langchain_astradb import AstraDBVectorStore
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from dotenv import load_dotenv
import os
from flipkart.data_converter import data_converter

load_dotenv()

GROQ_API = os.getenv("GROQ_API")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
HF_TOKEN = os.getenv("HF_TOKEN")

embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5")


def data_ingestion(status):
    vstore = AstraDBVectorStore(
        embedding=embeddings,
        collection_name="flipkart_bot",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_KEYSPACE,
    )

    storage = status
    if storage is None:
        docs = data_converter()
        insert_ids = vstore.add_documents(docs)
        return vstore, insert_ids
    else:
        return vstore, None  # Return None for insert_ids when status is "done"


if __name__ == "__main__":
    vstore, insert_ids = data_ingestion("done")

    if insert_ids:
        print(f"\nInserted {len(insert_ids)} documents")

    results = vstore.similarity_search("Can You tell me the low budget sound basshed?")
    print(results)
    for res in results:
        print(f"\n{res.page_content} [{res.metadata}]")
