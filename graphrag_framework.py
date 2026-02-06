import os
import pandas as pd
from langchain_groq import ChatGroq
import dotenv
import networkx as nx
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from PyPDF2 import PdfReader
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from prepare_graph import prepare_graph_for_llm
from chat_interface import ChatInterface

dotenv.load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

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

class GraphragFramework:
    def __init__(self):
        self.graph = None
        self.nodes = []
        self.edges = []
        self.node_types = []
        self.edge_types = []
        self.node_attributes = []
        self.edge_attributes = []
        self.node_attributes_types = []
        self.edge_attributes_types = []
        self.node_attributes_values = []
        self.edge_attributes_values = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)

    def add_node_type(self, node_type):
        self.node_types.append(node_type)

    def add_edge_type(self, edge_type):
        self.edge_types.append(edge_type)
    

class llm_graphrag_framework:
    def __init__(self, llm, prompt):
        self.graph = GraphragFramework()
        self.llm = llm
        self.prompt = prompt

    def generate_graph(self, prompt):
        return self.graph.generate_graph(prompt)

    def generate_node(self, prompt):
        return self.graph.generate_node(prompt)

    def generate_edge(self, prompt):
        return self.graph.generate_edge(prompt)
    







# Example usage
if __name__ == "__main__":
 
  
    chat_interface = ChatInterface(llm)
    chat_interface.run()