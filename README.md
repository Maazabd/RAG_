run at : https://ragapp477.streamlit.app/
# 📄 DocuSense RAG

A modern **Retrieval-Augmented Generation (RAG)** system powered by **ChromaDB** and **Groq API**, built with **Streamlit**. Extract insights from your PDF documents with AI-powered question answering and intelligent source citations.

## ✨ Features

- **📚 PDF Ingestion**: Upload and automatically parse PDF documents
- **🔍 Vector Search**: Semantic search across documents using ChromaDB
- **💬 AI Q&A**: Ask questions and get intelligent answers with source citations using Groq's LLaMA models
- **📊 Source References**: Expandable citations showing exact text excerpts from source documents
- **💡 Smart Suggestions**: AI-generated suggested questions based on your documents
- **🎨 Light Mode UI**: Clean, modern interface with smooth interactions
- **📱 Responsive Design**: Works seamlessly on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Conda or pip
- Groq API Key (get it free from [console.groq.com](https://console.groq.com))

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd path/to/rag
   ```

2. **Create a conda environment:**
   ```bash
   conda create -n rag_env python=3.11
   conda activate rag_env
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Groq API key:**
   - Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the application:**
   ```bash
   streamlit run main.py
   ```

   The app will open at `http://localhost:8501`

## 🌐 Deploy to Streamlit Cloud

### Deploy in 3 Steps:

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push -u origin main
   ```

2. **Create Streamlit Cloud App**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Select `main.py` as the entry point

3. **Add Secrets**
   - In Streamlit Cloud dashboard, click **Settings** → **Secrets**
   - Add your Groq API key:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

### Files Included for Cloud Deployment:
- `.streamlit/config.toml` - Streamlit configuration
- `requirements.txt` - All pinned dependencies
- `.gitignore` - Excludes `.env` and sensitive files

---

## 📖 How to Use

### 1. **Upload Documents**
   - Add PDF files to your project workspace directory
   - Click **"🚀 Ingest & Index PDFs Now"** to process them
   - Documents are parsed, chunked, and embedded in ChromaDB

### 2. **Browse Documents**
   - View all indexed documents in the left sidebar
   - Click a document name to open and view it in the PDF viewer
   - Use zoom controls to adjust viewing size

### 3. **Ask Questions**
   - Type your question in the query box
   - Or click any **💡 Suggested Question** for quick inspiration
   - Get AI-powered answers with source citations

### 4. **View Source References**
   - Click **"📄 View Source References"** to expand citations
   - See exact text excerpts with document name and page number
   - Understand where answers came from

### 5. **Manage Documents**
   - Use the trash icon (🗑) next to document names to delete them
   - Database automatically re-indexes when documents change

## 📁 Project Structure

```
rag/
├── main.py                      # Streamlit UI application
├── rag_engine.py               # RAG core engine (ChromaDB + Groq)
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (API keys)
├── README.md                    # This file
├── chroma_db/                   # ChromaDB vector database
│   └── (auto-generated database files)
└── suggested_questions.json     # Cached suggested questions
```

## 🛠️ Configuration

### Environment Variables (.env)

```env
GROQ_API_KEY=your_api_key_here
```

### Customization

**In `rag_engine.py`:**
- `EMBEDDING_MODEL`: Change embedding model (default: `all-MiniLM-L6-v2`)
- `GROQ_MODEL`: Change LLM model (default: `llama-3.3-70b-versatile`)
- `CHROMA_DB_DIR`: Database location
- Chunk size and overlap settings

## 📦 Dependencies

```
streamlit>=1.28.0
chromadb>=0.4.0
groq>=0.4.0
pypdf>=3.17.0
python-dotenv>=1.0.0
sentence-transformers>=2.2.0
numpy<2.0
```

## 🔧 Troubleshooting

### "No PDF files found"
- Ensure PDF files are in the project root directory
- Restart Streamlit after adding files

### "ChromaDB compatibility error"
```bash
pip install "numpy<2.0"
```

### "sentence_transformers not installed"
```bash
pip install sentence_transformers
```

### Groq API errors
- Verify API key is correct in `.env`
- Check [Groq Console](https://console.groq.com) for rate limits

### "SentenceTransformerEmbeddingFunction error" (Streamlit Cloud)
- This is handled automatically by environment variables set in `rag_engine.py`
- Ensure `.streamlit/config.toml` is in your repository
- Make sure all requirements are pinned in `requirements.txt`

## 📊 How It Works

### RAG Pipeline

```
PDF Files
    ↓
Extract Text (PyPDF)
    ↓
Split into Chunks (750 tokens, 150 overlap)
    ↓
Generate Embeddings (Sentence Transformers)
    ↓
Store in ChromaDB (Vector Database)
    ↓
User Query
    ↓
Generate Query Embedding
    ↓
Semantic Search (Find similar chunks)
    ↓
Groq LLaMA (Generate answer with context)
    ↓
Return Answer + Citations
```

## 🎯 Key Components

### **main.py**
- Streamlit UI with sidebar navigation
- Document viewer with PDF rendering
- Q&A interface with suggested questions
- Source reference viewer

### **rag_engine.py**
- PDF text extraction
- Text chunking with overlap
- ChromaDB integration
- Groq LLaMA API calls
- Citation matching and formatting

### **ChromaDB**
- Local vector database
- Semantic similarity search
- Document metadata storage

## 💡 Tips & Tricks

- **Better answers**: Provide context in your questions
- **Faster ingestion**: Split large PDFs into smaller files
- **Suggested questions**: Delete `suggested_questions.json` to regenerate
- **Clear database**: Remove `chroma_db/` folder to reset everything

## 🐛 Known Limitations

- PDF extraction quality depends on document format
- Suggested questions refresh when document count changes
- Single session handling (not multi-user)
- Large documents may take time to process

## 🚀 Future Enhancements

- [ ] Multi-document comparison
- [ ] Custom prompts and system instructions
- [ ] Chat history and session management
- [ ] Web search integration
- [ ] Support for more file formats (DOCX, TXT, etc.)
- [ ] Advanced filtering and search options
- [ ] User authentication and multi-user support

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments in `main.py` and `rag_engine.py`
3. Check Streamlit and ChromaDB documentation

## 🎨 UI Features

- **Light Mode**: Clean, modern interface optimized for readability
- **Responsive Design**: Adapts to different screen sizes
- **Expandable Sections**: Collapse/expand citations and details
- **Interactive Elements**: Clickable documents and suggestions
- **Real-time Feedback**: Success/error messages for all actions

---

**Built with ❤️ using Streamlit, ChromaDB, and Groq**
