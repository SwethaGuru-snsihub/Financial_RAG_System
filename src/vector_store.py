import os
import json
import pickle
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

class FinancialVectorStore:
    """FAISS-based vector store with financial context-aware retrieval."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        self.top_k = int(os.getenv('TOP_K_RETRIEVAL', 5))
        
        # Financial query expansion terms
        self.financial_synonyms = {
            'profit': ['net income', 'earnings', 'profit', 'income'],
            'revenue': ['revenue', 'sales', 'income', 'top line'],
            'expenses': ['expenses', 'costs', 'expenditures', 'spending'],
            'assets': ['assets', 'resources', 'holdings'],
            'debt': ['debt', 'liabilities', 'borrowings', 'obligations'],
            'cash flow': ['cash flow', 'cash generation', 'operating cash', 'free cash flow'],
            'margin': ['margin', 'profitability', 'profit margin', 'operating margin']
        }
    
    def create_index(self, embeddings: np.ndarray):
        """Create FAISS index from embeddings."""
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings.astype(np.float32))
        
        # Create FAISS index (Inner Product for cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        
        print(f"Created FAISS index with {self.index.ntotal} vectors")
    
    def load_chunks(self, chunks: List[Dict]):
        """Load chunk metadata."""
        self.chunks = chunks
        print(f"Loaded {len(self.chunks)} chunks")
    
    def save_index(self, output_dir: str):
        """Save FAISS index and metadata."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(output_dir, 'faiss_index.bin'))
        
        # Save chunks metadata
        with open(os.path.join(output_dir, 'chunks_metadata.pkl'), 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"Saved index and metadata to {output_dir}")
    
    def load_index(self, input_dir: str):
        """Load FAISS index and metadata."""
        # Load FAISS index
        index_path = os.path.join(input_dir, 'faiss_index.bin')
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Load chunks metadata
        metadata_path = os.path.join(input_dir, 'chunks_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"Loaded {len(self.chunks)} chunks metadata")
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with financial synonyms for better retrieval."""
        query_lower = query.lower()
        expanded_queries = [query]
        
        for term, synonyms in self.financial_synonyms.items():
            if term in query_lower:
                for synonym in synonyms:
                    if synonym != term:
                        expanded_query = query_lower.replace(term, synonym)
                        expanded_queries.append(expanded_query)
        
        return expanded_queries[:3]  # Limit to avoid too many queries
    
    def contextual_search(self, query: str, section_filter: Optional[str] = None, 
                         top_k: Optional[int] = None) -> List[Dict]:
        """Perform contextual search with financial awareness."""
        if self.index is None:
            raise ValueError("Index not created or loaded")
        
        top_k = top_k or self.top_k
        
        # Expand query for better retrieval
        expanded_queries = self.expand_query(query)
        
        all_results = []
        
        for expanded_query in expanded_queries:
            # Generate query embedding
            query_embedding = self.model.encode([expanded_query])
            query_embedding = query_embedding.astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            # Perform similarity search
            scores, indices = self.index.search(query_embedding, min(top_k * 2, len(self.chunks)))
            
            # Collect results with metadata
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    result = {
                        'chunk': chunk,
                        'score': float(score),
                        'query_used': expanded_query
                    }
                    all_results.append(result)
        
        # Remove duplicates and sort by score
        seen_chunks = set()
        unique_results = []
        
        for result in all_results:
            chunk_id = result['chunk']['id']
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_results.append(result)
        
        # Sort by score (descending)
        unique_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply section filter if provided
        if section_filter:
            unique_results = [
                r for r in unique_results 
                if r['chunk']['section_type'] == section_filter
            ]
        
        # Apply contextual re-ranking
        reranked_results = self._contextual_rerank(query, unique_results[:top_k * 2])
        
        return reranked_results[:top_k]
    
    def _contextual_rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """Re-rank results based on financial context and query intent."""
        query_lower = query.lower()
        
        # Financial context scoring
        for result in results:
            chunk = result['chunk']
            context_score = 0
            
            # Boost score for relevant section types
            if 'profit' in query_lower or 'income' in query_lower or 'earning' in query_lower:
                if chunk['section_type'] == 'income_statement':
                    context_score += 0.2
            
            elif 'asset' in query_lower or 'liability' in query_lower or 'equity' in query_lower:
                if chunk['section_type'] == 'balance_sheet':
                    context_score += 0.2
            
            elif 'cash' in query_lower and 'flow' in query_lower:
                if chunk['section_type'] == 'cash_flow':
                    context_score += 0.2
            
            # Boost score for chunks with numbers (likely financial data)
            if chunk['metadata'].get('has_numbers', False):
                context_score += 0.1
            
            # Boost score for chunks with tables
            if chunk['metadata'].get('has_tables', False):
                context_score += 0.1
            
            # Boost score for relevant financial keywords
            chunk_keywords = chunk['metadata'].get('financial_keywords', [])
            for keyword in chunk_keywords:
                if keyword in query_lower:
                    context_score += 0.05
            
            # Update final score
            result['final_score'] = result['score'] + context_score
        
        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return results
    
    def get_similar_chunks(self, chunk_id: int, top_k: int = 3) -> List[Dict]:
        """Get similar chunks to a given chunk."""
        if chunk_id >= len(self.chunks):
            return []
        
        target_chunk = self.chunks[chunk_id]
        query_text = target_chunk['text']
        
        # Search for similar chunks
        results = self.contextual_search(query_text, top_k=top_k + 1)
        
        # Remove the original chunk from results
        similar_chunks = [r for r in results if r['chunk']['id'] != chunk_id]
        
        return similar_chunks[:top_k]
    
    def get_section_chunks(self, section_type: str) -> List[Dict]:
        """Get all chunks from a specific section type."""
        section_chunks = [
            chunk for chunk in self.chunks 
            if chunk['section_type'] == section_type
        ]
        return section_chunks
    
    def search_by_keywords(self, keywords: List[str], top_k: Optional[int] = None) -> List[Dict]:
        """Search chunks by financial keywords."""
        top_k = top_k or self.top_k
        matching_chunks = []
        
        for chunk in self.chunks:
            chunk_text_lower = chunk['text'].lower()
            chunk_keywords = chunk['metadata'].get('financial_keywords', [])
            
            # Check for keyword matches
            keyword_matches = 0
            for keyword in keywords:
                if keyword.lower() in chunk_text_lower or keyword.lower() in chunk_keywords:
                    keyword_matches += 1
            
            if keyword_matches > 0:
                matching_chunks.append({
                    'chunk': chunk,
                    'keyword_matches': keyword_matches,
                    'score': keyword_matches / len(keywords)  # Normalize score
                })
        
        # Sort by number of keyword matches
        matching_chunks.sort(key=lambda x: x['keyword_matches'], reverse=True)
        
        return matching_chunks[:top_k]
    
    def get_statistics(self) -> Dict:
        """Get vector store statistics."""
        if not self.chunks:
            return {}
        
        section_counts = {}
        total_chunks = len(self.chunks)
        chunks_with_numbers = sum(1 for c in self.chunks if c['metadata'].get('has_numbers', False))
        chunks_with_tables = sum(1 for c in self.chunks if c['metadata'].get('has_tables', False))
        
        for chunk in self.chunks:
            section = chunk['section_type']
            section_counts[section] = section_counts.get(section, 0) + 1
        
        return {
            'total_chunks': total_chunks,
            'section_distribution': section_counts,
            'chunks_with_numbers': chunks_with_numbers,
            'chunks_with_tables': chunks_with_tables,
            'index_size': self.index.ntotal if self.index else 0
        }

if __name__ == "__main__":
    # Test the vector store
    vector_store = FinancialVectorStore()