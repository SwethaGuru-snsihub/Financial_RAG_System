"""
Main entry point for Financial RAG System
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_pipeline import FinancialRAGPipeline

def main():
    parser = argparse.ArgumentParser(description='Financial RAG System')
    parser.add_argument('--mode', choices=['cli', 'ui'], default='ui',
                       help='Run mode: cli for command line, ui for Streamlit interface')
    parser.add_argument('--pdf', type=str, help='Path to PDF file to process')
    parser.add_argument('--query', type=str, help='Question to ask (CLI mode only)')
    parser.add_argument('--pdfs', type=str, help='Comma-separated paths to multiple PDF files')
    
    args = parser.parse_args()
    
    if args.mode == 'ui':
        run_streamlit_app()
    else:
        run_cli_mode(args.pdf, args.query, args.pdfs)

def run_streamlit_app():
    """Launch Streamlit UI."""
    try:
        import streamlit.web.cli as stcli
        import sys
        
        # Set up the command to run streamlit
        sys.argv = ["streamlit", "run", "ui/app.py"]
        stcli.main()
    except ImportError:
        print("Streamlit not installed. Install with: pip install streamlit")
    except Exception as e:
        print(f"Error launching Streamlit: {e}")

def run_cli_mode(pdf_path, query, pdfs_path):
    """Run in command line mode."""
    
    print("🚀 Financial RAG System - CLI Mode")
    print("=" * 50)
    
    # Initialize RAG pipeline
    rag = FinancialRAGPipeline()
    
    # Check if we need to process a document
    if pdf_path:
        print(f"📄 Processing document: {pdf_path}")
        if not os.path.exists(pdf_path):
            print(f"❌ Error: File {pdf_path} not found")
            return
        
        success = rag.process_document(pdf_path)
        if not success:
            print("❌ Failed to process document")
            return
        print("✅ Document processed successfully!")
    
    elif pdfs_path:
        pdf_paths = [p.strip() for p in pdfs_path.split(',')]
        success = rag.process_multiple_documents(pdf_paths)
        if not success:
            print("❌ Failed to process multiple documents")
            return
        print("✅ Multiple documents processed successfully!")
    
    # Try to load existing processed document
    elif not rag.load_processed_document():
        print("❌ No processed document found. Please provide a PDF file with --pdf")
        return
    
    # Interactive mode or single query
    if query:
        # Single query mode
        print(f"\n❓ Question: {query}")
        print("-" * 50)
        
        result = rag.answer_question(query)
        print(f"📝 Answer:\n{result['answer']}")
        print(f"\n📊 Confidence: {result['confidence']:.1%}")
        
        if result['sources']:
            print(f"\n📚 Sources:")
            for i, source in enumerate(result['sources'][:3], 1):
                print(f"{i}. {source['section_type'].replace('_', ' ').title()} (Page {source['page_number']})")
    else:
        # Interactive mode
        print("\n💬 Interactive Q&A Mode")
        print("Type 'quit' to exit, 'help' for commands")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n❓ Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    show_help()
                    continue
                
                elif user_input.lower() == 'summary':
                    show_summary(rag)
                    continue
                
                elif not user_input:
                    continue
                
                # Process question
                print("🤔 Analyzing...")
                result = rag.answer_question(user_input)
                
                print(f"\n📝 Answer:\n{result['answer']}")
                print(f"\n📊 Confidence: {result['confidence']:.1%}")
                
                if result['sources']:
                    print(f"\n📚 Sources:")
                    for i, source in enumerate(result['sources'][:3], 1):
                        print(f"{i}. {source['section_type'].replace('_', ' ').title()} (Page {source['page_number']})")
                
                if result.get('followup_questions'):
                    print(f"\n💡 Suggested follow-up questions:")
                    for i, followup in enumerate(result['followup_questions'], 1):
                        print(f"{i}. {followup}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def show_help():
    """Show help information."""
    print("\n📖 Available Commands:")
    print("• help     - Show this help message")
    print("• summary  - Show document summary")
    print("• quit     - Exit the program")
    print("\n💡 Sample Questions:")
    print("• What was the net profit for 2023?")
    print("• What were the major operating expenses?")
    print("• How did the revenue change YoY?")
    print("• What is the company's debt-to-equity ratio?")
    print("• Show me cash flow from operations")

def show_summary(rag):
    """Show document summary."""
    try:
        summary = rag.get_document_summary()
        stats = summary['statistics']
        
        print("\n📋 Document Summary:")
        print(f"• Total chunks: {stats['total_chunks']}")
        print(f"• Chunks with numbers: {stats['chunks_with_numbers']}")
        print(f"• Chunks with tables: {stats['chunks_with_tables']}")
        
        print(f"\n📊 Section Distribution:")
        for section, count in stats['section_distribution'].items():
            percentage = (count / stats['total_chunks']) * 100
            print(f"• {section.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"❌ Error getting summary: {e}")

if __name__ == "__main__":
    main()