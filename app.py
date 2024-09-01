from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

import os 
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_TRACING_V2 = True
LANGCHAIN_PROJECT = 'Chat Bot'
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
groq_api = os.getenv('GROQ_API_KEY')

def my_func(question, model_name, temparature, max_tokens):
    model = ChatGroq(model=model_name, groq_api_key=groq_api, temperature=temparature, max_tokens=max_tokens)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "your the teacher for the user, so teach then what they asked in a more understandable way."),
            ("user", "{question}")
        ]
    )
    parser = StrOutputParser()
    chain = prompt|model|parser
    response = chain.invoke({"question": question})

    return response

st.title("Welcome to OpenSourse ChatBot \n")

st.sidebar.title("Settings")
model_name = st.sidebar.selectbox("Please Select the model: ", ["gemma2-9b-it", "gemma-7b-it", "llama3-groq-70b-8192-tool-use-preview", 
                                                   "llama3-groq-8b-8192-tool-use-preview", "llama-3.1-70b-versatile", 
                                                   "llama-3.1-8b-instant", "llama-guard-3-8b", 'llama3-70b-8192', "llama3-8b-8192",
                                                   "mixtral-8x7b-32768"])

temparature = st.sidebar.slider("Temparature", min_value=0.2, max_value=1.0, value=0.4)
max_tokens = st.sidebar.slider("Max Token", min_value=100, max_value=1000, value=500)

question = st.text_input("Enter the Query: ")
button = st.button("ask")

if button or question:
    if len(question)==0:
        st.html("<p style='text-align: right;'>Please ask the question!</p>")
    else:
        response = my_func(question, model_name, temparature, max_tokens)

        st.html(f"<div style='margin-left: 250px'> <p style='border-radius: 5px; text-align: center; padding: 5px; background-color: #3f444d;'>{question}</p></div>")
        st.write(response)