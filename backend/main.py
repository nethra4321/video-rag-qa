# backend/main.py
import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel

from backend.rag import process_video, ask_question

app = FastAPI()


class ProcessReq(BaseModel):
    video_path: str


class AskReq(BaseModel):
    video_id: str
    question: str
    model: str | None = None  


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/process")
def process(req: ProcessReq):
    return process_video(req.video_path)


@app.post("/ask")
def ask(req: AskReq):
    llama = os.getenv("OLLAMA_LLAMA", os.getenv("OLLAMA_LLAMA3", "llama3.2:3b"))
    qwen = os.getenv("OLLAMA_QWEN", "qwen2.5:3b-instruct")

    model_map = {
        "llama3": llama,
        "qwen": qwen,
        "gpt2": "__gpt2__",  
    }

    chosen = (req.model or "llama3").strip().lower()
    actual_model = model_map.get(chosen, llama)

    return ask_question(req.video_id, req.question, model=actual_model)
