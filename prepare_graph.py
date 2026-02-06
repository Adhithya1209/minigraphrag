import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class prepare_graph_for_llm:
    def __init__(self):
        self.graph = None

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

    def text_chunking(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(text)
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
        
    def add_chunk_ids_hierarchical(chunks, doc_id):
        """Create hierarchical IDs like 'doc1_chunk_0', 'doc1_chunk_1'"""
        
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = f"{doc_id}_chunk_{i}"
            chunk.metadata['doc_id'] = doc_id
            chunk.metadata['chunk_index'] = i
        
        return chunks