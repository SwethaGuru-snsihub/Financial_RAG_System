import streamlit as st
import os
import sys
import tempfile
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.rag_pipeline import FinancialRAGPipeline   

# Page configuration
st.set_page_config(
    page_title="Financial RAG System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = FinancialRAGPipeline()
    st.session_state.document_processed = False
    st.session_state.chat_history = []

def main():
    st.markdown('<h1 class="main-header">📊 Financial RAG System</h1>', unsafe_allow_html=True)
    st.markdown("**Analyze Financial Reports with AI-Powered Q&A**")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Multi-file upload
        uploaded_files = st.file_uploader(
            "Upload Multiple Financial Reports (PDF)",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload multiple annual financial reports"
        )

        if uploaded_files and st.button("🔄 Process Multiple Documents", type="primary"):
            file_paths = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    file_paths.append(tmp_file.name)
            success = st.session_state.rag_pipeline.process_multiple_documents(file_paths)
            # Clean up temp files if needed
            for path in file_paths:
                os.unlink(path)
            if success:
                st.success("✅ Multiple documents processed successfully!")
                st.rerun()
            else:
                st.error("❌ Error processing multiple documents.")

        # Document status
        st.header("📈 System Status")
        if st.session_state.document_processed:
            st.success("✅ Document processed and ready")
            
            # Show document summary
            if st.button("📋 View Document Summary"):
                show_document_summary()
        else:
            st.warning("⏳ Please upload and process a document")
        
        # Sample questions
        st.header("💡 Sample Questions")
        sample_questions = [
            "What was the net profit for 2023?",
            "What were the major operating expenses?", 
            "How did the revenue change YoY?",
            "What is the company's total debt?",
            "Show me the cash flow statement summary"
        ]
        
        for question in sample_questions:
            if st.button(f"❓ {question}", key=f"sample_{question}"):
                if st.session_state.document_processed:
                    st.session_state.current_question = question
                    st.rerun()
                else:
                    st.error("Please process a document first")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Chat interface
        question = st.text_input(
            "Enter your financial question:",
            placeholder="e.g., What was the revenue for 2023?",
            key="question_input"
        )
        
        # Handle sample question
        if hasattr(st.session_state, 'current_question'):
            question = st.session_state.current_question
            st.text_input(
                "Enter your financial question:",
                value=question,
                key="sample_question_display"
            )
            delattr(st.session_state, 'current_question')
        
        col_ask, col_clear = st.columns([1, 1])
        
        with col_ask:
            ask_button = st.button("🔍 Ask Question", type="primary", disabled=not st.session_state.document_processed)
        
        with col_clear:
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Process question
        if ask_button and question and st.session_state.document_processed:
            answer_question(question)
        
        # Display chat history
        display_chat_history()
    
    with col2:
        st.header("📊 Quick Metrics")
        
        if st.session_state.document_processed:
            # Quick metric searches
            metric_options = ["Revenue", "Profit", "Assets", "Debt", "Cash", "Margins"]
            selected_metric = st.selectbox("Select Financial Metric:", metric_options)
            
            if st.button(f"📈 Search {selected_metric}"):
                search_metric(selected_metric.lower())
            
            # Period comparison
            st.subheader("📅 Period Comparison")
            metric_for_comparison = st.text_input("Metric to compare:", placeholder="e.g., revenue")
            period1 = st.text_input("Period 1:", value="2023")
            period2 = st.text_input("Period 2:", value="2022")
            
            if st.button("⚖️ Compare Periods"):
                if metric_for_comparison:
                    compare_periods(metric_for_comparison, [period1, period2])

def process_document(uploaded_file):
    """Process uploaded PDF document."""
    try:
        with st.spinner("🔄 Processing document... This may take a few minutes."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Process document
            success = st.session_state.rag_pipeline.process_document(tmp_path)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            if success:
                st.session_state.document_processed = True
                st.success("✅ Document processed successfully!")
                st.balloons()
            else:
                st.error("❌ Error processing document. Please try again.")
                
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def answer_question(question):
    """Process question and display answer."""
    try:
        with st.spinner("🤔 Analyzing financial data..."):
            result = st.session_state.rag_pipeline.answer_question(question)
        
        # Add to chat history
        st.session_state.chat_history.append({
            'question': question,
            'result': result
        })
        
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error processing question: {str(e)}")

def display_chat_history():
    """Display chat history with answers and sources."""
    if not st.session_state.chat_history:
        st.info("💡 Ask a question about the financial report to get started!")
        return
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        question = chat['question']
        result = chat['result']
        
        with st.expander(f"Q: {question}", expanded=(i == 0)):
            # Display answer
            st.markdown("**📝 Answer:**")
            st.write(result['answer'])
            
            # Display confidence
            confidence = result['confidence']
            if confidence >= 0.7:
                conf_class = "confidence-high"
                conf_icon = "🟢"
            elif confidence >= 0.4:
                conf_class = "confidence-medium"
                conf_icon = "🟡"
            else:
                conf_class = "confidence-low"
                conf_icon = "🔴"
            
            st.markdown(f"{conf_icon} **Confidence:** <span class='{conf_class}'>{confidence:.1%}</span>", 
                       unsafe_allow_html=True)
            
            # Display sources
            if result['sources']:
                st.markdown("**📚 Sources:**")
                for j, source in enumerate(result['sources'][:3]):
                    with st.container():
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source {j+1}:</strong> {source['section_type'].replace('_', ' ').title()} 
                            (Page {source['page_number']})
                            <br><small>{source['text']}</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Display follow-up questions
            if result.get('followup_questions'):
                st.markdown("**🔍 Suggested Follow-up Questions:**")
                for followup in result['followup_questions']:
                    if st.button(f"❓ {followup}", key=f"followup_{i}_{followup}"):
                        st.session_state.current_question = followup
                        st.rerun()

def show_document_summary():
    """Display document processing summary."""
    try:
        summary = st.session_state.rag_pipeline.get_document_summary()
        
        st.header("📋 Document Summary")
        
        # Statistics
        stats = summary['statistics']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Chunks", stats['total_chunks'])
        with col2:
            st.metric("With Numbers", stats['chunks_with_numbers'])
        with col3:
            st.metric("With Tables", stats['chunks_with_tables'])
        with col4:
            st.metric("Index Size", stats['index_size'])
        
        # Section distribution
        st.subheader("📊 Section Distribution")
        section_dist = stats['section_distribution']
        
        for section, count in section_dist.items():
            percentage = (count / stats['total_chunks']) * 100
            st.write(f"**{section.replace('_', ' ').title()}:** {count} chunks ({percentage:.1f}%)")
        
        # Section samples
        st.subheader("📖 Section Samples")
        section_samples = summary['section_samples']
        
        for section, data in section_samples.items():
            with st.expander(f"{section.replace('_', ' ').title()} ({data['chunk_count']} chunks)"):
                st.write(data['sample_text'])
                
    except Exception as e:
        st.error(f"Error loading document summary: {str(e)}")

def search_metric(metric):
    """Search for specific financial metric."""
    try:
        with st.spinner(f"🔍 Searching for {metric} data..."):
            results = st.session_state.rag_pipeline.search_financial_metrics(metric)
        
        if results:
            st.success(f"Found {len(results)} results for {metric}")
            
            for i, result in enumerate(results[:3]):
                chunk = result['chunk']
                with st.expander(f"Result {i+1} - {chunk['section_type'].replace('_', ' ').title()}"):
                    st.write(chunk['text'])
                    st.caption(f"Page {chunk['page_number']} | Score: {result['score']:.2f}")
        else:
            st.warning(f"No results found for {metric}")
            
    except Exception as e:
        st.error(f"Error searching for {metric}: {str(e)}")

def compare_periods(metric, periods):
    """Compare metric across different periods."""
    try:
        with st.spinner(f"⚖️ Comparing {metric} across {', '.join(periods)}..."):
            comparison = st.session_state.rag_pipeline.compare_periods(metric, periods)
        
        if comparison['comparison_available']:
            st.success(f"Comparison data found for {metric}")
            
            for period, data in comparison['data'].items():
                st.subheader(f"📅 {period}")
                for result in data[:2]:
                    chunk = result['chunk']
                    st.write(f"**{chunk['section_type'].replace('_', ' ').title()}:** {chunk['text'][:300]}...")
        else:
            st.warning(f"Insufficient data for comparison of {metric}")
            
    except Exception as e:
        st.error(f"Error comparing periods: {str(e)}")

if __name__ == "__main__":
    main()