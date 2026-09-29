from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=1000) # type: ignore

st.title("LangChain Prompt UI")
# st.header("Enter your prompt below:")
# user_prompt = st.text_input("Enter your prompt here:", key="user_prompt")


# st.button("Submit", on_click=lambda: st.write(model.invoke(user_prompt).content))

paper_input = st.selectbox("Select a cuisine:", ["Italian", "Chinese", "Mexican", "Indian", "French"], key="paper_input")

style_input = st.selectbox("Select a style:", ["Casual", "Fine Dining", "Street Food", "Fusion", "Traditional"], key="style_input")

budget_input = st.slider("Select a budget range:", 10, 100, (20, 50), key="budget_input")

# template
prompt_template = ChatPromptTemplate.from_template("""
You are a professional food recommendation assistant.

User preferences:
- Cuisine: {paper_input}
- Dining style: {style_input}
- Budget range: ${budget_min} - ${budget_max}

Recommend a dish that fits these preferences. 

Explain:
- Dish name
- Description
- Why it fits
- Estimated price
""")

prompt = prompt_template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "budget_min": budget_input[0],
    "budget_max": budget_input[1]
})

st.button("Get Recommendation", on_click=lambda: st.write(model.invoke(prompt).content))


