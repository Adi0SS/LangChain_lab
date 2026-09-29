from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# This shit costs money bitch !!!!!!!!
model = ChatOpenAI(model="gpt-4")
result = model.invoke("")
print(result)
