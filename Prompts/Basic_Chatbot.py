from langchain_openai import ChatOpenAI   
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=1000) # type: ignore

while True:
    user_input = input("user: ")
    if user_input == "quit":
        break
    AI_response = model.invoke(user_input).content
    print(type(AI_response))
    print(f"AI: {AI_response}")
