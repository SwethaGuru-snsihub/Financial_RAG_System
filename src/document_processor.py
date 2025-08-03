import os
import re
import json
from typing import List, Dict, Tuple
import PyPDF2
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class FinancialDocumentProcessor:
    """Processes financial PDFs with context-aware chunking and embeddings."""
    
    def __init__(self):
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 500))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 50))
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Financial section patterns
        self.section_patterns = {
            'income_statement': [
                r'consolidated statements? of (?:operations|income)',
                r'income statement',
                r'profit and loss',
                r'statement of earnings'
            ],
            'balance_sheet': [
                r'consolidated balance sheets?',
                r'statement of financial position',
                r'balance sheet'
            ],
            'cash_flow': [
                r'consolidated statements? of cash flows?',
                r'cash flow statement',
                r'statement of cash flows?'
            ],
            'notes': [
                r'notes to (?:consolidated )?financial statements?',
                r'notes to the financial statements?'
            ]
        }
        
        # Financial keywords for enhanced chunking
        self.financial_keywords = [
            'revenue', 'net income', 'operating income', 'gross profit',
            'assets', 'liabilities', 'equity', 'cash', 'debt',
            'operating cash flow', 'free cash flow', 'capex',
            'earnings per share', 'dividend', 'margin'
        ]
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF with page and section metadata."""
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            pages = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                
                # Clean and normalize text
                text = self._clean_text(text)
                
                # Detect section type
                section_type = self._detect_section_type(text)
                
                pages.append({
                    'page_number': page_num + 1,
                    'text': text,
                    'section_type': section_type,
                    'char_count': len(text)
                })
        
        return pages
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR issues
        text = re.sub(r'(\d),(\d)', r'\1,\2', text)  # Fix number formatting
        text = re.sub(r'(\$)\s+(\d)', r'\1\2', text)  # Fix currency formatting
        
        return text.strip()
    
    def _detect_section_type(self, text: str) -> str:
        """Detect financial section type from text content."""
        text_lower = text.lower()
        
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return section_type
        
        return 'general'
    
    def create_contextual_chunks(self, pages: List[Dict]) -> List[Dict]:
        """Create context-aware chunks with financial metadata."""
        chunks = []
        chunk_id = 0
        
        for page in pages:
            text = page['text']
            section_type = page['section_type']
            page_num = page['page_number']
            
            # Split into sentences for better chunking
            sentences = self._split_into_sentences(text)
            
            current_chunk = ""
            current_sentences = []
            
            for sentence in sentences:
                # Check if adding sentence exceeds chunk size
                if len(current_chunk + sentence) > self.chunk_size and current_chunk:
                    # Create chunk with metadata
                    chunk_metadata = self._extract_chunk_metadata(
                        current_chunk, section_type, page_num, chunk_id
                    )
                    
                    chunks.append({
                        'id': chunk_id,
                        'text': current_chunk.strip(),
                        'page_number': page_num,
                        'section_type': section_type,
                        'metadata': chunk_metadata
                    })
                    
                    chunk_id += 1
                    
                    # Start new chunk with overlap
                    overlap_sentences = current_sentences[-2:] if len(current_sentences) >= 2 else current_sentences
                    current_chunk = " ".join(overlap_sentences) + " " + sentence
                    current_sentences = overlap_sentences + [sentence]
                else:
                    current_chunk += " " + sentence
                    current_sentences.append(sentence)
            
            # Add final chunk if it has content
            if current_chunk.strip():
                chunk_metadata = self._extract_chunk_metadata(
                    current_chunk, section_type, page_num, chunk_id
                )
                
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'page_number': page_num,
                    'section_type': section_type,
                    'metadata': chunk_metadata
                })
                chunk_id += 1
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with financial context awareness."""
        # Simple sentence splitting with financial number preservation
        sentences = re.split(r'(?<!\d)\.(?!\d)', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short fragments
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _extract_chunk_metadata(self, chunk_text: str, section_type: str, page_num: int, chunk_id: int) -> Dict:
        """Extract metadata from chunk for enhanced retrieval."""
        metadata = {
            'has_numbers': bool(re.search(r'\$?[\d,]+\.?\d*', chunk_text)),
            'has_tables': bool(re.search(r'(\$?[\d,]+\.?\d*\s*){3,}', chunk_text)),
            'financial_keywords': [],
            'table_number': None
        }
        
        # Identify financial keywords
        chunk_lower = chunk_text.lower()
        for keyword in self.financial_keywords:
            if keyword in chunk_lower:
                metadata['financial_keywords'].append(keyword)
        
        # Detect table references
        table_match = re.search(r'table (\d+)', chunk_lower)
        if table_match:
            metadata['table_number'] = int(table_match.group(1))
        
        return metadata
    
    def generate_embeddings(self, chunks: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """Generate embeddings for chunks using sentence transformers."""
        
        # Prepare texts for embedding
        texts_for_embedding = []
        enhanced_chunks = []
        
        for chunk in chunks:
            # Enhance text with section context for better embeddings
            enhanced_text = self._enhance_text_for_embedding(chunk)
            texts_for_embedding.append(enhanced_text)
            enhanced_chunks.append(chunk)
        
        # Generate embeddings
        embeddings = self.model.encode(texts_for_embedding, show_progress_bar=True)
        
        return embeddings, enhanced_chunks
    
    def _enhance_text_for_embedding(self, chunk: Dict) -> str:
        """Enhance chunk text with context for better embeddings."""
        text = chunk['text']
        section_type = chunk['section_type']
        
        # Add section context as prefix
        section_prefixes = {
            'income_statement': 'Financial performance and earnings: ',
            'balance_sheet': 'Financial position and assets: ',
            'cash_flow': 'Cash flow and liquidity: ',
            'notes': 'Financial notes and details: '
        }
        
        prefix = section_prefixes.get(section_type, 'Financial information: ')
        return prefix + text
    
    def save_processed_data(self, chunks: List[Dict], embeddings: np.ndarray, output_dir: str):
        """Save processed chunks and embeddings."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save chunks as JSON
        with open(os.path.join(output_dir, 'chunks.json'), 'w') as f:
            json.dump(chunks, f, indent=2)
        
        # Save embeddings as numpy array
        np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings)
        
        print(f"Processed {len(chunks)} chunks and saved to {output_dir}")
    
    def process_document(self, pdf_path: str, output_dir: str = 'data/processed') -> Tuple[List[Dict], np.ndarray]:
        """Complete document processing pipeline."""
        print(f"Processing document: {pdf_path}")
        
        # Extract text from PDF
        pages = self.extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(pages)} pages")
        
        # Create contextual chunks
        chunks = self.create_contextual_chunks(pages)
        print(f"Created {len(chunks)} contextual chunks")
        
        # Generate embeddings
        embeddings, enhanced_chunks = self.generate_embeddings(chunks)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Save processed data
        self.save_processed_data(enhanced_chunks, embeddings, output_dir)
        
        return enhanced_chunks, embeddings

if __name__ == "__main__":
    # Test the processor
    processor = FinancialDocumentProcessor()