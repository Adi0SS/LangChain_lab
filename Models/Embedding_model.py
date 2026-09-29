from langchain_openai import OpenAIEmbeddings 
from dotenv import load_dotenv

load_dotenv()
# This shit costs money !!!!!!!!
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)


result = embedding_model.embed_query("My name is adarsh.")

print(str(result))