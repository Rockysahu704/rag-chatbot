import streamlit as st
import requests

st.title("RAG Chatbot")

# Multiple PDF Upload
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# Upload PDFs
if uploaded_files:

    for uploaded_file in uploaded_files:

        files = {
            "file": uploaded_file
        }

        response = requests.post(
            "http://127.0.0.1:8000/upload",
            files=files
        )

        st.success(response.json()["message"])

# Question Input
query = st.text_input("Ask Question")

# Ask Question
if st.button("Ask"):

    with st.spinner("Generating answer..."):

        response = requests.get(
            "http://127.0.0.1:8000/ask",
            params={"query": query}
        )

        data = response.json()

        # Final Answer
        st.subheader("Answer")
        st.write(data["answer"])

        # Sources
        st.subheader("Sources")

        for source in data["sources"]:

            st.write(
                f"{source['filename']} — Page {source['page']}"
            )

        # Retrieved Chunks
        st.subheader("Retrieved Chunks")

        for chunk in data["retrieved_chunks"]:

            st.info(chunk["text"])

            st.write(
                f"Source: {chunk['filename']} | "
                f"Page: {chunk['page']}"
            )

            if "score" in chunk:
                st.write(f"Similarity Score: {chunk['score']}")

            st.divider()