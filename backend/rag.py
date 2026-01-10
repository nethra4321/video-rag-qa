# backend/rag.py
"""
Video RAG 
- ffmpeg extracts audio
- faster-whisper transcribes (timestamps)
- sentence-transformers embeds chunks
- faiss stores vectors
- cross-encoder reranks retrieved chunks
- Ollama (LLaMA/Qwen) generates final grounded answer
- GPT-2 baseline optional
- Soft prompting + Map-Reduce for broad/distributed questions
- Adds Proof lines (direct quotes + timestamps)

Fixes in this version:
- Soft prompt is used for RETRIEVAL embedding too (better candidates for vague questions).
- Map step no longer asks the model to output [#] tags (we append citations ourselves).
- Light cleanup to avoid generic filler like "thanks for watching".
"""

import os
import json
import subprocess
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import faiss
import requests
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# Storage
INDEX_DIR = os.getenv("INDEX_DIR", os.path.join("data", "indexes"))
os.makedirs(INDEX_DIR, exist_ok=True)

#  Models (Embeddings and Rerank)
EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("HF_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
embedder = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

# whisper settings
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")

# Retrieval / Chunking
TOP_K = int(os.getenv("TOP_K", "30"))
RERANK_K = int(os.getenv("RERANK_K", "12"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "3"))

# ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_LLAMA = os.getenv("OLLAMA_LLAMA", os.getenv("OLLAMA_LLAMA3", "llama3.2:3b"))
OLLAMA_QWEN = os.getenv("OLLAMA_QWEN", "qwen2.5:3b-instruct")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))

try:
    from transformers import pipeline
except Exception:
    pipeline = None

_gpt2_pipe = None


@dataclass
class Chunk:
    text: str
    start: float
    end: float


def _index_paths(video_id: str):
    idx_path = os.path.join(INDEX_DIR, f"{video_id}.faiss")
    meta_path = os.path.join(INDEX_DIR, f"{video_id}.json")
    return idx_path, meta_path


def _extract_audio(video_path: str, out_wav: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", out_wav],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _transcribe(audio_wav: str) -> List[Tuple[str, float, float]]:
    wm = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    segments, _info = wm.transcribe(audio_wav, vad_filter=True)

    out: List[Tuple[str, float, float]] = []
    for seg in segments:
        txt = (seg.text or "").strip()
        if txt:
            out.append((txt, float(seg.start), float(seg.end)))
    return out


def _chunk_segments(
    segments: List[Tuple[str, float, float]],
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    i = 0
    while i < len(segments):
        start = segments[i][1]
        end = segments[i][2]
        buf: List[str] = []
        j = i

        while j < len(segments):
            t, _s, e = segments[j]
            candidate = (" ".join(buf + [t])).strip()
            if len(candidate) > max_chars and buf:
                break
            buf.append(t)
            end = e
            j += 1

        chunks.append(Chunk(text=" ".join(buf).strip(), start=start, end=end))
        i = max(i + 1, j - overlap)

    return chunks


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


def _embed(texts: List[str]) -> np.ndarray:
    vecs = embedder.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    vecs = vecs.astype("float32")
    return _normalize(vecs)


def process_video(video_path: str) -> dict:
    if not os.path.exists(video_path):
        return {"ok": False, "error": f"Video not found: {video_path}"}

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    idx_path, meta_path = _index_paths(video_id)
    wav_path = os.path.join(INDEX_DIR, f"{video_id}__audio.wav")

    try:
        _extract_audio(video_path, wav_path)
        segments = _transcribe(wav_path)
        chunks = _chunk_segments(segments)

        texts = [c.text for c in chunks]
        if not texts:
            return {"ok": False, "error": "No speech detected (empty transcript)."}

        vecs = _embed(texts)
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        faiss.write_index(index, idx_path)

        meta = {
            "video_id": video_id,
            "video_path": video_path,
            "chunks": [{"text": c.text, "start": c.start, "end": c.end} for c in chunks],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return {"ok": True, "video_id": video_id, "chunks_indexed": len(chunks)}

    finally:
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except OSError:
            pass


def _load(video_id: str):
    idx_path, meta_path = _index_paths(video_id)
    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        return None, None
    index = faiss.read_index(idx_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


def _format_ts(seconds: float) -> str:
    s = int(seconds)
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _ollama_is_up() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _length_plan(video_seconds: float) -> tuple[int, int]:
    # keep bounded to reduce timeouts
    if video_seconds <= 60:
        return 160, 280
    if video_seconds <= 5 * 60:
        return 260, 380
    if video_seconds <= 15 * 60:
        return 320, 450
    return 380, 600


def _ollama_generate(prompt: str, model: Optional[str], num_predict: int) -> str:
    use_model = (model or OLLAMA_LLAMA).strip()

    payload = {
        "model": use_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "num_predict": num_predict,
            # Helps reduce rambling / outro lines
            "repeat_penalty": 1.1,
        },
    }

    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=(5, OLLAMA_TIMEOUT))

    if r.status_code != 200:
        try:
            err = r.json().get("error", r.text)
        except Exception:
            err = r.text
        raise requests.HTTPError(f"Ollama /api/generate failed ({r.status_code}): {err}", response=r)

    return (r.json().get("response") or "").strip()


def _gpt2_generate(prompt: str, max_new_tokens: int = 180) -> str:
    """
    GPT-2 has a HARD limit (1024 tokens). We must truncate prompt.
    """
    global _gpt2_pipe
    if pipeline is None:
        raise RuntimeError("GPT-2 requires: pip install transformers torch")

    if _gpt2_pipe is None:
        base = os.getenv("GPT2_BASE_MODEL", "gpt2")
        _gpt2_pipe = pipeline("text-generation", model=base)

    tok = _gpt2_pipe.tokenizer
    max_ctx = getattr(tok, "model_max_length", 1024)
    max_prompt_tokens = max(1, max_ctx - max_new_tokens - 5)

    ids = tok.encode(prompt, add_special_tokens=False)
    if len(ids) > max_prompt_tokens:
        ids = ids[-max_prompt_tokens:]
        prompt = tok.decode(ids)

    out = _gpt2_pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        truncation=True,
    )
    text = out[0]["generated_text"]
    return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()


def _soft_prompt(question: str) -> str:
    q = (question or "").strip()
    ql = q.lower()
    broad_phrases = [
        "what are they talking about",
        "what is this video about",
        "summarize",
        "summary",
        "explain this video",
        "overall",
        "main points",
    ]
    if len(q.split()) <= 7 or any(p in ql for p in broad_phrases):
        return "Summarize the main topics discussed and the key points made across the video."
    return q


def map_Reduce(question: str, ranked_chunks: List[Dict[str, Any]], video_seconds: float, rr_scores: List[float]) -> bool:
    q = (question or "").strip()
    broad_q = len(q.split()) <= 7

    if not ranked_chunks:
        return False

    time_span = float(ranked_chunks[-1]["end"]) - float(ranked_chunks[0]["start"])
    spread_out = (video_seconds > 0) and (time_span > 0.35 * video_seconds)

    flat_scores = False
    if rr_scores:
        flat_scores = (max(rr_scores) - min(rr_scores)) < 0.15

    return broad_q or (spread_out and flat_scores)


def _make_proof_lines(ranked: List[Dict[str, Any]], max_lines: int = 6) -> List[str]:
    proof = []
    for i, c in enumerate(ranked[:max_lines], start=1):
        txt = (c.get("text") or "").strip().replace("\n", " ")
        if len(txt) > 220:
            txt = txt[:220].rstrip() + "…"
        proof.append(f'- [{i}] ({_format_ts(c["start"])}-{_format_ts(c["end"])}) "{txt}"')
    return proof


_FALLBACK_BAD_LINES = re.compile(
    r"(thank(s)? you for watching|like and subscribe|subscribe to|hit the bell|follow for more)",
    re.IGNORECASE,
)


def _clean_summary(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("<summary>", "").replace("</summary>", "").strip()
    s = _FALLBACK_BAD_LINES.sub("", s).strip()
    # remove stray bracket artifacts like [#]
    s = s.replace("[#]", "").strip()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ask_question(video_id: str, question: str, model: Optional[str] = None) -> dict:
    index, meta = _load(video_id)
    if (index, meta) == (None, None):
        return {"ok": False, "error": "Index not found. Click Process Video first."}

    if model != "__gpt2__" and not _ollama_is_up():
        return {"ok": False, "error": "Ollama is not running. Start it with: `ollama serve`."}

    chunks: List[Dict[str, Any]] = meta["chunks"]
    video_seconds = float(chunks[-1]["end"]) if chunks else 0.0
    _target_words, num_predict = _length_plan(video_seconds)

    # soft prompt for retrieval and generation
    soft_q = _soft_prompt(question)

    # 1) retrieve (embed soft_q, not raw question)
    qv = _embed([soft_q])
    _scores, ids = index.search(qv, TOP_K)

    candidates: List[Dict[str, Any]] = []
    for i in ids[0].tolist():
        if 0 <= i < len(chunks):
            candidates.append(chunks[i])

    if not candidates:
        return {"ok": False, "error": "No candidates retrieved from index."}

    # 2) rerank (safe fallback)
    pairs = [(soft_q, c["text"]) for c in candidates]
    try:
        rr_scores = reranker.predict(pairs).tolist()
        ranked_pairs = sorted(zip(candidates, rr_scores), key=lambda x: x[1], reverse=True)
    except Exception:
        rr_scores = [0.0] * len(candidates)
        ranked_pairs = list(zip(candidates, rr_scores))

    ranked_pairs = ranked_pairs[:min(RERANK_K, len(ranked_pairs))]
    ranked = [c for (c, _s) in ranked_pairs]
    ranked_scores = [float(s) for (_c, s) in ranked_pairs]

    # 3) context + evidence
    context_lines = []
    evidence = []
    for k, c in enumerate(ranked, start=1):
        context_lines.append(f"[{k}] ({_format_ts(c['start'])}-{_format_ts(c['end'])}) {c['text']}")
        evidence.append(
            {
                "start": c["start"],
                "end": c["end"],
                "start_ts": _format_ts(c["start"]),
                "end_ts": _format_ts(c["end"]),
            }
        )
    context = "\n".join(context_lines)
    proof_lines = _make_proof_lines(ranked, max_lines=6)

    # 4) choose mode 
    use_mr = map_Reduce(question, ranked, video_seconds, ranked_scores)
    if model == "__gpt2__":
        use_mr = True

    try:
        if use_mr:
            # map
            map_prompt_template = f"""Summarize the snippet in 1–2 sentences.

Rules:
- Use ONLY the snippet content.
- Do NOT add citations or bracket tags.
- Do NOT add filler like "thank you for watching".
- Output only the summary text.

Task: {soft_q}

Snippet:
{{SNIPPET}}

Summary:
"""
            mapped: List[str] = []
            for idx, snip in enumerate(context_lines[:6], start=1):
                mp = map_prompt_template.replace("{SNIPPET}", snip)

                if model == "__gpt2__":
                    summ = _gpt2_generate(mp, max_new_tokens=80)
                else:
                    summ = _ollama_generate(mp, model=model, num_predict=140)

                summ = _clean_summary(summ)
                if summ:
                    mapped.append(f"{summ} [{idx}]")

            if not mapped:
                return {"ok": False, "error": "Map step produced empty summaries. Try another model or shorter question."}

            # reduce
            if model == "__gpt2__":
                reduce_prompt = f"""Question: {soft_q}

Notes (each line ends with its source tag):
{chr(10).join(mapped)}

Write a short answer in 4–6 sentences using ONLY these notes.
"""
                core_answer = _gpt2_generate(reduce_prompt, max_new_tokens=160).strip()
                core_answer = core_answer.rstrip() + "\n\nProof:\n" + "\n".join(proof_lines) + "\n\nEvidence:\n" + ", ".join(
                    [f"[{i}]" for i in range(1, min(7, len(ranked) + 1))]
                )
            else:
                reduce_prompt = f"""You will answer using ONLY the snippet summaries.

Rules:
- Write a detailed, multi-paragraph answer.
- Each paragraph MUST include citations like [1], [2] based on the summary tags.
- Do NOT mention ambiguity/uncertainty/missing info.
- Include a Proof section with direct quotes from the context snippets.

Question: {soft_q}

Snippet summaries:
{chr(10).join(mapped)}

Output format:
Answer:
<answer>

Proof:
- [#] (timestamps) "quote"
- ...

Evidence:
[list citations used, like [1], [2], ...]
"""
                core_answer = _ollama_generate(reduce_prompt, model=model, num_predict=420).strip()

        else:
            # direct
            prompt = f"""You answer using ONLY the provided context snippets.

Rules:
- Write a detailed, multi-paragraph answer.
- Each paragraph MUST include citations like [1], [2] based on the snippets used in that paragraph.
- If the question is broad, give a best-effort summary of the discussion.
- Do NOT mention ambiguity/uncertainty/missing info.
- Include a Proof section with direct quotes from the context.
- No emojis.

Question: {soft_q}

Context:
{context}

Output format:
Answer:
<answer>

Proof:
- [#] (timestamps) "quote"
- ...

Evidence:
[list citations used, like [1], [2], ...]
"""
            core_answer = _ollama_generate(prompt, model=model, num_predict=num_predict).strip()

    except requests.HTTPError as e:
        return {"ok": False, "error": f"Ollama error: {str(e)}. Try pulling model: `ollama pull {model or OLLAMA_LLAMA}`"}
    except Exception as e:
        return {"ok": False, "error": f"LLM call failed: {str(e)}"}

    if not core_answer.strip():
        return {"ok": False, "error": "Model returned empty response. Check logs/model."}

    # Guarantee proof lines exist
    if "Proof:" not in core_answer:
        core_answer = core_answer.rstrip() + "\n\nProof:\n" + "\n".join(proof_lines) + "\n\nEvidence:\n" + ", ".join(
            [f"[{i}]" for i in range(1, min(7, len(ranked) + 1))]
        )

    return {
        "ok": True,
        "video_id": video_id,
        "answer": core_answer,
        "proof": proof_lines,
        "evidence": evidence,
        "llm": (model or OLLAMA_LLAMA),
        "seconds": video_seconds,
        "mode": "map-reduce" if use_mr else "direct",
    }
