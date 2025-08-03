import os
import json
import re
from typing import List, Dict, Optional, Tuple
import google.generativeai as genai
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from src.document_processor import FinancialDocumentProcessor

from .vector_store import FinancialVectorStore

load_dotenv()

class FinancialRAGPipeline:
    """Complete RAG pipeline for financial document Q&A with Gemini LLM."""
    
    def __init__(self):
        # Initialize components
        self.document_processor = FinancialDocumentProcessor()
        self.vector_store = FinancialVectorStore()
        
        # Configure Gemini
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # LLM configuration
        self.temperature = float(os.getenv('TEMPERATURE', 0.1))
        self.max_tokens = int(os.getenv('MAX_TOKENS', 1024))
        
        # Financial prompting templates
        self.setup_prompts()
    
    def setup_prompts(self):
        """Setup financial domain-specific prompts."""
        
        # Query classification prompt
        self.query_classifier_prompt = PromptTemplate(
            template="""
Classify this financial query into one of these categories:
- profit_loss: Questions about revenue, expenses, net income, earnings
- balance_sheet: Questions about assets, liabilities, equity, financial position  
- cash_flow: Questions about cash operations, investing, financing activities
- ratios: Questions about financial ratios, margins, and metrics
- comparison: Year-over-year or period comparisons
- general: Other financial questions

Query: {query}

Respond with only the category name.
""",
            input_variables=["query"]
        )
        
        # Main RAG response prompt with chain-of-thought
        self.rag_response_prompt = PromptTemplate(
            template="""
You are a financial analyst AI specializing in Fortune 500 company analysis. Use the provided financial document context to answer the question accurately and comprehensively.

<FINANCIAL_CONTEXT>
{context}
</FINANCIAL_CONTEXT>

<SOURCE_REFERENCES>
{source_refs}
</SOURCE_REFERENCES>

Question: {question}

Instructions for Analysis:
1. **Data Location**: Identify which financial section contains the answer
2. **Numerical Extraction**: Extract exact figures and verify accuracy
3. **Context Analysis**: Provide business context, trends, and insights
4. **Source Attribution**: Always cite the specific section and page number

Analysis Framework:
- First, determine the financial statement type (Income Statement, Balance Sheet, Cash Flow)
- Then, locate the specific line items or data points
- Provide quantitative data with proper formatting
- Include relevant context about trends, comparisons, or notable items
- Always cite sources in format: [Section: section_type, Page: page_number]

Financial Response Guidelines:
- Use precise financial terminology
- Format numbers with appropriate units (millions, billions, etc.)
- Highlight year-over-year changes when relevant
- Include percentage calculations when appropriate
- If data is unavailable, clearly state this limitation

Provide a comprehensive financial analysis with proper source citations:
""",
            input_variables=["context", "source_refs", "question"]
        )
        
        # Follow-up question generation prompt
        self.followup_prompt = PromptTemplate(
            template="""
Based on the financial analysis provided, suggest 3 relevant follow-up questions that would provide deeper insights into the company's financial performance.

Original Question: {original_question}
Analysis Provided: {analysis}

Generate 3 specific, actionable follow-up questions:
""",
            input_variables=["original_question", "analysis"]
        )
    
    def classify_query(self, query: str) -> str:
        """Classify the financial query to improve retrieval."""
        try:
            prompt = self.query_classifier_prompt.format(query=query)
            response = self.model.generate_content(prompt)
            classification = response.text.strip().lower()
            
            valid_categories = ['profit_loss', 'balance_sheet', 'cash_flow', 'ratios', 'comparison', 'general']
            if classification in valid_categories:
                return classification
            return 'general'
        except Exception as e:
            print(f"Error in query classification: {e}")
            return 'general'
    
    def process_document(self, pdf_path: str) -> bool:
        """Process a financial document and create vector store."""
        try:
            # Process document
            chunks, embeddings = self.document_processor.process_document(pdf_path)
            
            # Create and populate vector store
            self.vector_store.create_index(embeddings)
            self.vector_store.load_chunks(chunks)
            
            # Save for future use
            self.vector_store.save_index('data/processed')
            
            print(f"Successfully processed document: {pdf_path}")
            return True
        except Exception as e:
            print(f"Error processing document: {e}")
            return False
    
    def load_processed_document(self, data_dir: str = 'data/processed') -> bool:
        """Load previously processed document."""
        try:
            self.vector_store.load_index(data_dir)
            print("Successfully loaded processed document")
            return True
        except Exception as e:
            print(f"Error loading processed document: {e}")
            return False
    
    def process_multiple_documents(self, pdf_paths: list) -> bool:
        """Process multiple financial documents and create a combined vector store."""
        all_chunks = []
        all_embeddings = []

        try:
            for pdf_path in pdf_paths:
                chunks, embeddings = self.document_processor.process_document(pdf_path)
                all_chunks.extend(chunks)
                all_embeddings.append(embeddings)

            import numpy as np
            combined_embeddings = np.vstack(all_embeddings)

            self.vector_store.create_index(combined_embeddings)
            self.vector_store.load_chunks(all_chunks)
            self.vector_store.save_index('data/processed_multi')

            print(f"Successfully processed {len(pdf_paths)} documents.")
            return True
        except Exception as e:
            print(f"Error processing multiple documents: {e}")
            return False

    def load_processed_documents(self, data_dir: str = 'data/processed_multi') -> bool:
        """Load previously processed multiple documents."""
        try:
            self.vector_store.load_index(data_dir)
            print("Successfully loaded processed documents")
            return True
        except Exception as e:
            print(f"Error loading processed documents: {e}")
            return False
    
    def retrieve_context(self, query: str, top_k: int = 5) -> Tuple[List[Dict], str]:
        """Retrieve relevant context for the query."""
        
        # Classify query to determine section focus
        query_category = self.classify_query(query)
        
        # Map category to section filter
        section_mapping = {
            'profit_loss': 'income_statement',
            'balance_sheet': 'balance_sheet', 
            'cash_flow': 'cash_flow'
        }
        
        section_filter = section_mapping.get(query_category)
        
        # Retrieve relevant chunks
        results = self.vector_store.contextual_search(
            query=query,
            section_filter=section_filter,
            top_k=top_k
        )
        
        # Format context and source references
        context_parts = []
        source_refs = []
        
        for i, result in enumerate(results, 1):
            chunk = result['chunk']
            
            # Format context
            context_parts.append(f"Context {i}:\n{chunk['text']}")
            
            # Format source reference
            source_ref = f"Source {i}: Section={chunk['section_type']}, Page={chunk['page_number']}"
            if chunk['metadata'].get('table_number'):
                source_ref += f", Table={chunk['metadata']['table_number']}"
            source_refs.append(source_ref)
        
        context = "\n\n".join(context_parts)
        source_references = "\n".join(source_refs)
        
        return results, context, source_references
    
    def generate_response(self, query: str, context: str, source_refs: str) -> str:
        """Generate response using Gemini with financial context."""
        try:
            prompt = self.rag_response_prompt.format(
                context=context,
                source_refs=source_refs,
                question=query
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            
            return response.text
        except Exception as e:
            return f"Error generating response: {e}"
    
    def answer_question(self, query: str, include_sources: bool = True) -> Dict:
        """Complete Q&A pipeline."""
        
        # Retrieve relevant context
        results, context, source_refs = self.retrieve_context(query)
        
        if not results:
            return {
                'answer': "I couldn't find relevant information to answer your question. Please ensure the document has been processed and try rephrasing your question.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Generate response
        answer = self.generate_response(query, context, source_refs)
        
        # Calculate confidence based on retrieval scores
        avg_score = sum(r['final_score'] for r in results) / len(results)
        confidence = min(avg_score * 2, 1.0)  # Normalize to 0-1 range
        
        # Prepare source information
        sources = []
        if include_sources:
            for result in results:
                chunk = result['chunk']
                sources.append({
                    'text': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                    'section_type': chunk['section_type'],
                    'page_number': chunk['page_number'],
                    'score': result['final_score'],
                    'table_number': chunk['metadata'].get('table_number')
                })
        
        # Generate follow-up questions
        try:
            followup_prompt = self.followup_prompt.format(
                original_question=query,
                analysis=answer[:500]  # Truncate for prompt
            )
            followup_response = self.model.generate_content(followup_prompt)
            followup_questions = self._parse_followup_questions(followup_response.text)
        except:
            followup_questions = []
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': confidence,
            'query_category': self.classify_query(query),
            'followup_questions': followup_questions
        }
    
    def _parse_followup_questions(self, response: str) -> List[str]:
        """Parse follow-up questions from LLM response."""
        questions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered or bulleted questions
            if re.match(r'^\d+\.', line) or line.startswith('- '):
                question = re.sub(r'^\d+\.\s*|-\s*', '', line).strip()
                if question.endswith('?') and len(question) > 10:
                    questions.append(question)
        
        return questions[:3]  # Limit to 3 questions
    
    def get_document_summary(self) -> Dict:
        """Get summary of processed document."""
        stats = self.vector_store.get_statistics()
        
        # Get sample content from each section
        section_samples = {}
        for section_type in ['income_statement', 'balance_sheet', 'cash_flow', 'notes']:
            section_chunks = self.vector_store.get_section_chunks(section_type)
            if section_chunks:
                section_samples[section_type] = {
                    'chunk_count': len(section_chunks),
                    'sample_text': section_chunks[0]['text'][:200] + "..."
                }
        
        return {
            'statistics': stats,
            'section_samples': section_samples,
            'processing_status': 'ready' if self.vector_store.index else 'not_processed'
        }
    
    def search_financial_metrics(self, metric_type: str) -> List[Dict]:
        """Search for specific financial metrics."""
        
        metric_keywords = {
            'revenue': ['revenue', 'sales', 'net sales', 'total revenue'],
            'profit': ['net income', 'profit', 'earnings', 'net profit'],
            'assets': ['total assets', 'current assets', 'non-current assets'],
            'debt': ['total debt', 'long-term debt', 'short-term debt'],
            'cash': ['cash', 'cash equivalents', 'operating cash flow'],
            'margins': ['gross margin', 'operating margin', 'net margin']
        }
        
        keywords = metric_keywords.get(metric_type.lower(), [metric_type])
        results = self.vector_store.search_by_keywords(keywords)
        
        return results
    
    def compare_periods(self, metric: str, periods: List[str] = ['2023', '2022']) -> Dict:
        """Compare financial metrics across different periods."""
        
        all_results = []
        for period in periods:
            query = f"{metric} {period}"
            results, _, _ = self.retrieve_context(query, top_k=3)
            all_results.extend(results)
        
        # Group results by period
        period_data = {}
        for result in all_results:
            chunk_text = result['chunk']['text']
            for period in periods:
                if period in chunk_text:
                    if period not in period_data:
                        period_data[period] = []
                    period_data[period].append(result)
        
        return {
            'metric': metric,
            'periods': periods,
            'data': period_data,
            'comparison_available': len(period_data) >= 2
        }

if __name__ == "__main__":
    # Test the RAG pipeline
    rag = FinancialRAGPipeline()
    
    print("Financial RAG Pipeline ready!")
    print("Available methods:")
    print("- process_document(pdf_path)")
    print("- load_processed_document()")
    print("- answer_question(query)")
    print("- get_document_summary()")
    print("- search_financial_metrics(metric_type)")