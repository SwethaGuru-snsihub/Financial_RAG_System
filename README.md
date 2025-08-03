## 📊 Financial RAG System
A Retrieval-Augmented Generation (RAG) system for analyzing Fortune 500 financial reports using AI-powered Q&A capabilities.

## 🏗️ Architecture

**RAG Type**: Modular/Contextual RAG with financial domain specialization

### Technical Stack
- **Framework**: LangChain (Agentic AI orchestration)
- **Vector DB**: FAISS (IndexFlatIP for cosine similarity)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **LLM**: Google Gemini any model
- **Chunking**: Section-aware with financial table preservation
- **UI**: Streamlit

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Google API Key (for Gemini)

### Installation Steps

1. **Clone the repository**
bash
git clone <repository-url>
cd financial-rag-system


2. **Install dependencies**
bash
pip install -r requirements.txt


3. **Set up environment variables**
bash
cp .env.example .env
# Edit .env and add your Gemini API key


Required environment variables:
bash
GEMINI_API_KEY=your_gemini_api_key_here

4. **Create data directories**
bash
mkdir -p data/raw data/processed

## 🚀 Usage

### Streamlit Web Interface

bash
python main.py --mode ui

or
bash
streamlit run ui/app.py


**Features:**
- Drag & drop PDF upload
- Interactive Q&A interface
- Source highlighting and citations
- Financial metrics search
- Period comparison tools
- Document summary dashboard

## 📖 Sample Questions

### Financial Performance
- "What was the net profit for 2023?"
- "How did the revenue change year-over-year?"
- "What were the major operating expenses?"
- "What is the gross profit margin?"

### Financial Position  
- "What is the company's total debt?"
- "What are the current assets and liabilities?"
- "What is the debt-to-equity ratio?"
- "How much cash does the company have?"

### Cash Flow Analysis
- "What was the operating cash flow?"
- "How much did the company spend on capital expenditures?"
- "What was the free cash flow?"

- "Did the company pay dividends?"
