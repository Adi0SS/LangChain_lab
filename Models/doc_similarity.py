from langchain_openai import OpenAIEmbeddings 
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
# This shit costs money !!!!!!!!
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

doc_ = [
    "American former professional boxer",
    "Born June 30, 1966, in Brooklyn, New York",
    "Known by the nickname 'Iron Mike'",
    "Became the youngest heavyweight boxing champion in history at age 20",
    "Won the WBC heavyweight title in 1986",
    "Later unified the WBC, WBA, and IBF heavyweight titles",
    "Known for exceptional punching power and aggressive fighting style",
    "Famous for his peek-a-boo defensive style",
    "Trained under legendary boxing trainer Cus D'Amato",
    "Became a professional boxer in 1985",
    "Won 50 professional fights",
    "Won 44 fights by knockout",
    "Suffered 6 professional losses",
    "Retired from professional boxing in 2005",
    "Returned to boxing for an exhibition match in 2020",
    "One of the most recognizable figures in boxing history",
    "Known for his explosive combinations and powerful hooks",
    "Inducted into the International Boxing Hall of Fame",
]

query = "How many professional fights did Mike Tyson win?"

doc_embeddings = embedding_model.embed_documents(doc_)

query_embedding = embedding_model.embed_query(query)

result = cosine_similarity(np.array([query_embedding]), np.array(doc_embeddings))[0]
index, score = sorted(list(enumerate(result)), key=lambda x: x[1], reverse=True)[0]
print(query)
print(doc_[index])
print(score)

# Top-k results
k = 3

top_k_indices = sorted(list(enumerate(result)), key=lambda x: x[1], reverse=True)[:k]
for index, score in top_k_indices:
    print(doc_[index])
    print(score)
