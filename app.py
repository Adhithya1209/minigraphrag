import os
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import dotenv
import streamlit as st
from chat_interface import ChatInterface

dotenv.load_dotenv()



if __name__ == "__main__":
    groq_api_key = os.getenv("GROQ_API_KEY")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    llm = ChatOllama(
            model="llama3.2",
            temperature=0,
            base_url="http://localhost:11434",  # Default Ollama URL
            # Optional parameters
            num_predict=2048,  # Max tokens to generate
            top_k=10,
            top_p=0.9,
        )
        

    # Set page configuration
    st.set_page_config(
        page_title="Chat with Graphrag",
        page_icon="📁",
        layout="wide"
    )

    # Title and description
    st.title("📁 Chat with Graphrag Interface")
    st.markdown("Upload and process different types of files with Graphrag")
    max_tokens = st.number_input(
            "Max Tokens",
            min_value=100,
            max_value=8000,
            value=1024
        )

    chat_interface = ChatInterface(llm)
    chat_interface.run()