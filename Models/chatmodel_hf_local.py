from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()
LLM = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm = LLM)
result = model.invoke("who is the president of Armenia?")
# print(result.content)
#  this takes a lot of cpu processing power and might end up crashing


