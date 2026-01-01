import os
import requests
import streamlit as st

# Backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Folder where videos are stored
VIDEO_DIR = os.getenv("VIDEO_DIR", os.path.join("data", "videos"))

# Page setup
st.set_page_config(page_title="Video RAG", layout="wide")
st.title("Video RAG")

# Check if backend is running
def backend_running():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False

# Safe POST request to backend
def post_to_backend(path, payload, timeout=300):
    r = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

# Stop if video folder does not exist
if not os.path.exists(VIDEO_DIR):
    st.error("Video folder not found")
    st.stop()

# List video files
videos = [v for v in os.listdir(VIDEO_DIR) if v.lower().endswith(".mp4")]
if not videos:
    st.info("Add mp4 files to the videos folder")
    st.stop()

left, right = st.columns([1, 2])

# left panel
with left:
    st.subheader("Select Video")
    selected = st.selectbox("Video file", videos)
    video_path = os.path.join(VIDEO_DIR, selected)
    video_id = os.path.splitext(selected)[0]

    st.subheader("Backend Status")
    if backend_running():
        st.write("Backend running")
    else:
        st.write("Backend not running")

    st.subheader("Process Video")
    if st.button("Process Video", disabled=not backend_running()):
        with st.spinner("Processing video"):
            result = post_to_backend(
                "/process",
                {"video_path": os.path.abspath(video_path)},
                timeout=1200
            )
        st.success(result)

    st.subheader("Ask Question")
    question = st.text_input("Enter your question")

    if st.button("Ask", disabled=not backend_running() or not question):
        with st.spinner("Generating answer"):
            result = post_to_backend(
                "/ask",
                {"video_id": video_id, "question": question}
            )
        st.session_state["answer"] = result

# right panel
with right:
    st.subheader("Video")
    st.video(video_path)

    st.subheader("Output")
    result = st.session_state.get("answer")

    if result:
        st.write("Answer:")
        st.write(result.get("answer", ""))

        st.write("Evidence:")
        for e in result.get("evidence", []):
            st.write(f"{e['start']} - {e['end']}")
