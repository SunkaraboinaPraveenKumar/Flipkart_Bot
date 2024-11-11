import pandas as pd
from langchain_core.documents import Document


def data_converter():
    product_data = pd.read_csv("C:/Users/sunka/PycharmProjects/Flipkart_Chatbot/data/flipkart_reviews_dataset.csv")
    data = product_data[['product_title', 'review']]

    product_list = []
    for index, row in data.iterrows():
        object_ = {
            "product_name": row['product_title'],
            "review": row['review']
        }
        product_list.append(object_)

    docs = []
    for object_ in product_list:
        metadata = {"product_name": object_['product_name']}
        page_content = object_["review"]
        doc = Document(page_content=page_content, metadata=metadata)
        docs.append(doc)

    docs = [[float(x) for x in doc if isinstance(x, (int, float))] for doc in docs]
    reconstructed_docs = []
    for i, vector in enumerate(docs):
        # Retrieve metadata from the corresponding original product
        metadata = {"product_name": product_list[i]['product_name']}

        # Convert the vector back to string format for page_content if needed
        page_content = " ".join(map(str, vector))  # or any other format that suits `page_content`

        # Recreate Document
        doc = Document(page_content=page_content, metadata=metadata)
        reconstructed_docs.append(doc)

    return reconstructed_docs



