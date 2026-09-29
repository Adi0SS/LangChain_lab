from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from huggingface_hub import model_info

load_dotenv()
llm = HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-V4-Pro-0813:fireworks-ai",
    task="text_generation",
    temperature=0,
    repetition_penalty=1.15,
    
    
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("what is a sextant")
print(result)
