# ui/app.py
import hashlib
import os
import re

import requests
import streamlit as st


# -------------------- Configuration --------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.getenv(
    "VIDEO_DIR",
    os.path.join(BASE_DIR, "data", "videos"),
)
os.makedirs(VIDEO_DIR, exist_ok=True)

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000",
).rstrip("/")

MODEL_OPTIONS = {
    "GPT-5": "gpt5",
    "LLaMA 3.2": "llama3",
    "Qwen 2.5": "qwen",
    "GPT-2 Baseline": "gpt2",
}


# -------------------- Page --------------------

st.set_page_config(
    page_title="Video RAG",
    page_icon="🎥",
    layout="wide",
)

st.title("Video RAG")
st.caption(
    "Upload a video, choose a language model, and ask questions "
)


# -------------------- Helpers --------------------

def backend_running() -> bool:
    try:
        response = requests.get(
            f"{BACKEND_URL}/health",
            timeout=3,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def post_to_backend(
    path: str,
    payload: dict,
    timeout: int = 300,
) -> dict:
    try:
        response = requests.post(
            f"{BACKEND_URL}{path}",
            json=payload,
            timeout=timeout,
        )
    except requests.Timeout as error:
        raise RuntimeError(
            "The request took too long. Please try again."
        ) from error
    except requests.ConnectionError as error:
        raise RuntimeError(
            "Unable to connect to the backend. "
            "Make sure FastAPI is running."
        ) from error

    if response.status_code >= 400:
        try:
            body = response.json()
            message = (
                body.get("detail")
                or body.get("error")
                or response.text
            )
        except Exception:
            message = response.text

        raise RuntimeError(message)

    return response.json()


def safe_filename(filename: str) -> str:
    name = os.path.basename(filename)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)

    if not name.lower().endswith(".mp4"):
        name += ".mp4"

    return name


def file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def save_uploaded_video(
    filename: str,
    file_bytes: bytes,
) -> str:
    safe_name = safe_filename(filename)
    stem, extension = os.path.splitext(safe_name)
    short_hash = file_hash(file_bytes)[:12]

    destination = os.path.join(
        VIDEO_DIR,
        f"{stem}_{short_hash}{extension}",
    )

    # Avoid rewriting the same file on every Streamlit rerun.
    if not os.path.exists(destination):
        with open(destination, "wb") as file:
            file.write(file_bytes)

    return os.path.abspath(destination)


def reset_video_state() -> None:
    st.session_state["processed_video_id"] = None
    st.session_state["processing_result"] = None
    st.session_state["answer"] = None


# -------------------- Session state --------------------

if "uploaded_file_hash" not in st.session_state:
    st.session_state["uploaded_file_hash"] = None

if "processed_video_id" not in st.session_state:
    st.session_state["processed_video_id"] = None

if "processing_result" not in st.session_state:
    st.session_state["processing_result"] = None

if "answer" not in st.session_state:
    st.session_state["answer"] = None

if "selected_model_label" not in st.session_state:
    st.session_state["selected_model_label"] = "GPT-5"


# -------------------- Upload --------------------

uploaded_video = st.file_uploader(
    "Upload an MP4 video",
    type=["mp4"],
    accept_multiple_files=False,
)

video_path = None
current_video_hash = None

if uploaded_video is not None:
    uploaded_bytes = uploaded_video.getvalue()
    current_video_hash = file_hash(uploaded_bytes)

    # New video selected
    if current_video_hash != st.session_state["uploaded_file_hash"]:
        reset_video_state()
        st.session_state["uploaded_file_hash"] = current_video_hash

    try:
        video_path = save_uploaded_video(
            uploaded_video.name,
            uploaded_bytes,
        )
    except OSError as error:
        st.error(f"Unable to save the uploaded video: {error}")
        st.stop()


# -------------------- Automatic processing --------------------

if uploaded_video is not None and video_path is not None:
    if not backend_running():
        st.error(
            "The application backend is unavailable. "
            "Start FastAPI and refresh this page."
        )
        st.stop()

    if st.session_state.get("processed_video_id") is None:
        with st.status(
            "Preparing your video...",
            expanded=True,
        ) as status:
            try:
                st.write("Extracting audio")
                st.write("Transcribing speech")
                st.write("Creating searchable video chunks")

                processing_result = post_to_backend(
                    "/process",
                    {
                        "video_path": video_path,
                    },
                    timeout=1800,
                )

                if not processing_result.get("ok"):
                    raise RuntimeError(
                        processing_result.get(
                            "error",
                            "Video processing failed.",
                        )
                    )

                st.session_state["processed_video_id"] = (
                    processing_result.get("video_id")
                )
                st.session_state["processing_result"] = (
                    processing_result
                )

                if processing_result.get("already_processed"):
                    status.update(
                        label="Video ready — existing index reused",
                        state="complete",
                        expanded=False,
                    )
                else:
                    status.update(
                        label="Video ready",
                        state="complete",
                        expanded=False,
                    )

            except Exception as error:
                status.update(
                    label="Processing failed",
                    state="error",
                    expanded=True,
                )
                st.error(str(error))
                st.stop()


# -------------------- Main layout --------------------

if uploaded_video is None:
    st.info("Upload an MP4 video to begin.")
    st.stop()

left, right = st.columns(
    [1.05, 1.95],
    gap="large",
)


# -------------------- Left panel --------------------

with left:
    st.subheader("Ask about the video")

    processing_result = st.session_state.get(
        "processing_result"
    )

    if processing_result:
        chunk_count = processing_result.get(
            "chunks_indexed",
            0,
        )

        st.success(
            "video ready"
        )

    with st.form("question_form"):
        selected_model_label = st.radio(
            "Model",
            options=list(MODEL_OPTIONS.keys()),
            index=list(MODEL_OPTIONS.keys()).index(
                st.session_state["selected_model_label"]
            ),
        )
        st.session_state["selected_model_label"] = selected_model_label

        question = st.text_area(
            "Question",
            placeholder="What is this video about?",
            height=110,
        )

        ask_submitted = st.form_submit_button(
            "Ask question",
            type="primary",
            use_container_width=True,
        )

    if ask_submitted:
        processed_video_id = st.session_state.get("processed_video_id")

        if not question.strip():
            st.warning("Enter a question first.")
        elif not processed_video_id:
            st.warning("The video is still being prepared. Please wait.")
        else:
            selected_model = MODEL_OPTIONS[
                selected_model_label
            ]

            with st.spinner(
                f"Generating answer with "
                f"{selected_model_label}..."
            ):
                try:
                    answer_result = post_to_backend(
                        "/ask",
                        {
                            "video_id": processed_video_id,
                            "question": question.strip(),
                            "model": selected_model,
                        },
                        timeout=1200,
                    )

                    st.session_state["answer"] = answer_result

                except Exception as error:
                    st.session_state["answer"] = {
                        "ok": False,
                        "error": str(error),
                    }


# -------------------- Right panel --------------------

with right:
    st.subheader("Video")
    st.video(uploaded_video)

    result = st.session_state.get("answer")

    if result:
        st.divider()

        if not result.get("ok", False):
            st.error(
                result.get(
                    "error",
                    "Unable to generate an answer.",
                )
            )

        else:
            model_name = result.get("llm", "")
            mode = result.get("mode", "")

            st.caption(
                f"Model: {'GPT-5' if model_name == '__gpt5__' else model_name}"
            )

            st.subheader("Answer")
            st.markdown(result.get("answer", ""))

            evidence = result.get("evidence", [])

            if evidence:
                with st.expander(
                    "Evidence timestamps",
                    expanded=False,
                ):
                    for index, item in enumerate(
                        evidence,
                        start=1,
                    ):
                        st.write(
                            f"**[{index}]** "
                            f"{item.get('start_ts', '')} – "
                            f"{item.get('end_ts', '')}"
                        )

