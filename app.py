import os
import pandas as pd
from langchain_groq import ChatGroq
import dotenv
import streamlit as st
from chat_interface import ChatInterface

dotenv.load_dotenv()


# Example usage
if __name__ == "__main__":
    groq_api_key = os.getenv("GROQ_API_KEY")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

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