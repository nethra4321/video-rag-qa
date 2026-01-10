# Video RAG 

Ask questions about video content and get timestamped answers using a Video RAG pipeline.  
The system works by:
- Transcribing video audio into text
- Chunking transcripts semantically
- Indexing chunks using FAISS for vector similarity search
- Retrieving relevant chunks for a user’s question
- Generating grounded answers using LLMs (LLaMA-3, Qwen) with timestamps
- Serving everything through a FastAPI backend

### Docker set up
```bash
git clone https://github.com/nethra/video-rag-qa.git
cd video-rag-qa
docker-compose up
```

#### Backend
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # Configure environment variables
uvicorn backend.main:app --reload
```

Runs at `http://localhost:8000`

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

Runs at `http://localhost:3000`

---

### Environment Variables

Create `.env` in root:
```env
OLLAMA_HOST=http://localhost:11434
MODEL_NAME=llama3
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=300
```

---

### Requirements

- Python 3.9+
- Node.js 18+
