from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

def chat_model():
    load_dotenv()
    small_model, large_model = ChatGroq(model = "qwen/qwen3.8-27b", reasoning_effort="none", reasoning_format = "parsed"), ChatOpenAI(model = "gpt-5.4-mini")
    return small_model, large_model
