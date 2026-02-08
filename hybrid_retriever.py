import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
from graphrag_schema import GraphRAGSchema

class PrepareRetrieval(GraphRAGSchema):
    def __init__(self, llm):
        self.graph = None
        self.llm = llm
    def clean_pdf_text(self, text):
        # Remove excessive whitespace and newlines
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines to single
        text = re.sub(r' +', ' ', text)     # Multiple spaces to single
        
        # Fix hyphenated words split across lines
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Remove standalone newlines (keep paragraph breaks)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        
        # Clean up common artifacts
        text = text.replace('\x00', '')  # Null bytes
        text = text.replace('\uf0b7', '•')  # Bullet points
        
        # Strip leading/trailing whitespace
        text = text.strip()
    
        return text

    def text_chunking(self, text, doc_id):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(text)
        chunks = self.add_chunk_ids_hierarchical(chunks, doc_id)
        return chunks

    def metadata_extractor(self, reader):
        pdf_meta = reader.metadata or {}
        metadata = {
            'title': pdf_meta.get('/Title', '').strip(),
            'author': pdf_meta.get('/Author', '').strip(),
            'date': pdf_meta.get('/CreationDate', '').strip(),
            'subject': pdf_meta.get('/Subject', '').strip(),
            'keywords': pdf_meta.get('/Keywords', '').strip()
        }

        if not metadata['title'] or metadata['author']:
            first_page_text = reader.pages[0].extract_text()
            parsed_meta = self.parse_first_page(first_page_text)
            if not metadata['title']:
                metadata['title'] = parsed_meta.get('title', '')
            if not metadata['author']:
                metadata['author'] = parsed_meta.get('author', '')
            if not metadata['date']:
                metadata['date'] = parsed_meta.get('date', '')
            metadata['date'] = self.clean_pdf_date(metadata['date'])
        
        return metadata
    def parse_first_page(self, text):
        """Extract title, author, date from first page text"""
        metadata = {}
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Title is usually the first non-empty line or largest text
        # For most papers, it's within first 5 lines
        if lines:
            metadata['title'] = lines[0]
        
        # Find authors - common patterns
        author_patterns = [
            r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:,\s*[A-Z][a-z]+\s+[A-Z][a-z]+)*)',  # John Doe, Jane Smith
            r'([A-Z]\.\s*[A-Z][a-z]+(?:,\s*[A-Z]\.\s*[A-Z][a-z]+)*)',  # J. Doe, A. Smith
            r'(?:Authors?:?\s*)([^\n]+)',  # "Author: Name" or "Authors: Names"
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, text[:1000])  # Search first 1000 chars
            if match:
                metadata['author'] = match.group(1).strip()
                break
        
        # Find dates - various formats
        date_patterns = [
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',  # 15 January 2024
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})',  # January 15, 2024
            r'(\d{4}-\d{2}-\d{2})',  # 2024-01-15
            r'(\d{1,2}/\d{1,2}/\d{4})',  # 01/15/2024
            r'(?:Published|Date|Received):\s*([^\n]+)',  # "Published: date"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text[:2000])
            if match:
                metadata['date'] = match.group(1).strip()
                break
        
        return metadata

    def clean_pdf_date(self, date_string):
        """Convert PDF date format to readable format"""
        if not date_string:
            return ''
        
        # PDF date format: D:20240115120000+01'00'
        pdf_date_match = re.match(r"D:(\d{4})(\d{2})(\d{2})", date_string)
        if pdf_date_match:
            year, month, day = pdf_date_match.groups()
            try:
                return f"{year}-{month}-{day}"
            except:
                return date_string
        
        return date_string
        
    def add_chunk_ids_hierarchical(self, chunks, doc_id):
        """Create hierarchical IDs like 'doc1_chunk_0', 'doc1_chunk_1'"""
        
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = f"{doc_id}_chunk_{i}"
            chunk.metadata['doc_id'] = doc_id
            chunk.metadata['chunk_index'] = i
        
        return chunks
    
    def process_pdf(self, reader, doc_id):
        for page in reader.pages:
            raw_text = page.extract_text() + "\n"
            cleaned_text = self.prepare_graph_for_llm.clean_pdf_text(raw_text)
            full_text += cleaned_text + "\n"
            metadata = self.metadata_extractor(reader)
            print(metadata)
            docs = [Document(page_content=cleaned_text, metadata=metadata)]
            chunks = self.text_chunking(docs, doc_id)

    def chunk_embedding_generation(self, chunks):
        embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",  
        encode_kwargs={'normalize_embeddings': True})
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
        return vectorstore

    def rag_retrieval(self, vectorstore, query):
        retrieved_docs = vectorstore.similarity_search(query, k=5)
        return retrieved_docs
    
    def entity_extraction(self, chunk, chunk_metadata):
        prompt = f"""
        Extract entities and relationships from the following text.
        
        Instructions:
        - Identify entities (author, organizations, concepts, locations, technologies, etc.)
        - Extract relationships between entities
        - Return results in JSON format
        
        Text:
        {chunk}
        
        Return JSON with this exact structure:
        {{
            "entities": [
                {{"name": "Entity Name", "type": "AUTHOR|ORGANIZATION|CONCEPT|LOCATION|TECHNOLOGY", "description": "brief description"}},
            ],
            "relationships": [
                {{"source": "Entity1", "target": "Entity2", "relationship": "relationship_type", "description": "context"}},
            ]
        }}
        
        Only return valid JSON, no additional text.
        """
        try:
            messages = [
            SystemMessage(content="You are an expert at extracting structured information from text. Always return valid JSON."),
            HumanMessage(content=f"Extract entities from this text: {prompt}")
        ]
            response = self.llm.invoke(messages)
            raw_output = response.content
            
        except Exception as e:
            print(f"Groq API error: {e}")
            return {"entities": [], "relationships": []}
        
        try:
            json_text = raw_output.strip()
            if json_text.startswith("```json"):
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif json_text.startswith("```"):
                json_text = json_text.split("```")[1].split("```")[0].strip()
            
            extracted_data = json.loads(json_text)
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return {"entities": [], "relationships": []}
        # Enrich with metadata
        entities = []
        for entity in extracted_data.get("entities", []):
            entities.append({
                "name": entity.get("name", "").strip(),
                "type": entity.get("type", "CONCEPT").upper(),
                "description": entity.get("description", ""),
                "source_chunk_id": chunk_metadata.get("chunk_id"),
                "source_doc_id": chunk_metadata.get("doc_id")
            })
        
        relationships = []
        for rel in extracted_data.get("relationships", []):
            relationships.append({
                "source": rel.get("source", "").strip(),
                "target": rel.get("target", "").strip(),
                "relationship": rel.get("relationship", "RELATED_TO").upper(),
                "description": rel.get("description", ""),
                "source_chunk_id": chunk_metadata.get("chunk_id"),
                "source_doc_id": chunk_metadata.get("doc_id")
            })
        
        return {
            "entities": entities,
            "relationships": relationships
        }
    
