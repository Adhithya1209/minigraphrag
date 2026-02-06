import re
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from PyPDF2 import PdfReader
from prepare_graph import prepare_graph_for_llm
from langchain_core.documents import Document
import pandas as pd

class ChatInterface:
    def __init__(self, llm):
        """
        Initialize the chat interface
        
        Args:
            llm: Pre-configured LangChain LLM instance (ChatGroq)
        """
        self.prepare_graph_for_llm = prepare_graph_for_llm()
        # Initialize session state for messages if not exists
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Initialize session state for uploaded file
        if "uploaded_file" not in st.session_state:
            st.session_state.uploaded_file = None
        
        # Initialize session state for file content
        if "file_content" not in st.session_state:
            st.session_state.file_content = None
            
        self.llm = llm
   
        self.system_prompt = "You are a helpful assistant that can answer questions about the uploaded file."
        
    def upload_file(self):
        """
        Display file uploader and process uploaded file
        Returns the uploaded file object
        """
        st.sidebar.header("📁 File Upload")
        
        file = st.sidebar.file_uploader(
            "Upload a document", 
            type=["txt", "pdf", "docx", "xlsx", "csv"],
            help="Upload a file to ask questions about it"
        )
        
        if file and file != st.session_state.uploaded_file:
            # New file uploaded
            st.session_state.uploaded_file = file
            
            # Process file content based on type
            try:
                if file.type == "text/plain":
                    st.session_state.file_content = file.read().decode("utf-8")
                elif file.type == "text/csv":
                    
                    df = pd.read_csv(file)
                    st.session_state.file_content = df.to_string()
                # Add more file type handling as needed
                elif file.type == "application/pdf":
                    
                    pdf_reader = PdfReader(file)
                    full_text = ""
                    for page in pdf_reader.pages:
                        raw_text = page.extract_text() + "\n"
                        cleaned_text = self.prepare_graph_for_llm.clean_pdf_text(raw_text)
                        full_text += cleaned_text + "\n"
                        metadata = self.prepare_graph_for_llm.metadata_extractor(pdf_reader)
                        print(metadata)
                        docs = [Document(page_content=cleaned_text, metadata=metadata)]
                        self.prepare_graph_for_llm.text_chunking(docs)

                    st.session_state.file_content = full_text[:200]
                # Add system message about file upload
                st.session_state.messages.append({
                    "role": "system", 
                    "content": f"✅ File uploaded: {file.name}"
                })
                
                st.sidebar.success(f"✅ Uploaded: {file.name}")
                st.sidebar.info(f"Size: {file.size / 1024:.2f} KB")
                
                # TODO: Process file with GraphRAG
                # self.graphrag_framework.process_file(st.session_state.file_content)
                
            except Exception as e:
                st.sidebar.error(f"Error processing file: {str(e)}")
                
        elif file:
            st.sidebar.success(f"✅ Current file: {file.name}")
            
        return file

    def display_chat_history(self):
        """
        Display all previous messages in the chat
        """
        for message in st.session_state.messages:
            # Skip system messages (file upload notifications)
            if message["role"] == "system":
                continue
                
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def get_rag_context(self, user_query):
        """
        Get relevant context from uploaded file using RAG
        
        Args:
            user_query: The user's question
            
        Returns:
            str: Relevant context from the file
        """
        # TODO: Implement RAG retrieval using GraphragFramework
        # For now, return the full file content or first 2000 chars
        if st.session_state.file_content:
            # context = self.graphrag_framework.retrieve(user_query)
            # return context
            
            # Temporary: Return truncated file content
            max_chars = 2000
            if len(st.session_state.file_content) > max_chars:
                return st.session_state.file_content[:max_chars] + "..."
            return st.session_state.file_content
        return None

    def chat(self):
        """
        Main chat interface - handles user input and displays responses
        """
        # Display chat history
        self.display_chat_history()
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the uploaded file..."):
            
            # Check if file is uploaded
            if not st.session_state.uploaded_file:
                st.warning("⚠️ Please upload a file first!")
                return
            
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                self.generate_response(prompt)

    def generate_response(self, user_prompt):
        """
        Generate AI response using LLM with RAG context
        
        Args:
            user_prompt: The user's question
        """
        message_placeholder = st.empty()
        
        try:
            # Get relevant context from file using RAG
            rag_context = self.get_rag_context(user_prompt)
            
            # Prepare messages for LLM
            messages = []
            
            # Add system prompt with file context
            if rag_context:
                enhanced_system_prompt = f"""{self.system_prompt}

Here is the relevant content from the uploaded file:

{rag_context}

Use this information to answer the user's question accurately."""
                messages.append(SystemMessage(content=enhanced_system_prompt))
            else:
                messages.append(SystemMessage(content=self.system_prompt))
            
            # Add chat history (last 5 exchanges to manage context)
            recent_messages = [msg for msg in st.session_state.messages 
                             if msg["role"] != "system"][-10:]
            
            for msg in recent_messages:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            
            # Get response from LLM
            with st.spinner("🤔 Thinking..."):
                response = self.llm.invoke(messages)
                full_response = response.content
            
            # Display response
            message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response
            })
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Make sure your API key is valid and you have credits.")

    def run(self):
        """
        Main method to run the entire chat interface
        """
        # Upload file in sidebar
        self.upload_file()
        
        # Display chat interface
        self.chat()
        
        # Sidebar: Chat statistics
        if st.session_state.messages:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📊 Chat Statistics")
            user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
            ai_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
            st.sidebar.write(f"👤 Questions asked: {user_msgs}")
            st.sidebar.write(f"🤖 Responses: {ai_msgs}")
            
            # Clear chat button
            if st.sidebar.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()