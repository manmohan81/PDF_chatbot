import streamlit as st
import fitz  # PyMuPDF
import os
import tempfile
import requests
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set Streamlit page config
st.set_page_config(page_title="📄 PDF ChatBot with Ollama", layout="wide")
st.title("📄 Chat with your PDF using Ollama 2")

# Sidebar options
option = st.sidebar.radio("Choose PDF Source:", ["Upload PDF", "GitHub URL"])
##st.sidebar.markdown("---")
##st.sidebar.markdown(
##    "<div style='text-align: center; font-size: 15px; margin-top: 5px;'>"
##    "Made by <strong>Manmohan Singh Rawat</strong></div>",
####    unsafe_allow_html=True
##)
st.sidebar.markdown(
    """
    <style>
    .bottom-footer {
        position: fixed;
        bottom: 15px;
        left: 0;
        width: 18rem;
        text-align: center;
        font-size: 13px;
        color: #888;
    }
    </style>
    <div class="bottom-footer">Made by <b>MANMOHAN SINGH RAWAT</b></div>
    """,
    unsafe_allow_html=True
)

pdf_path = None

if option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

elif option == "GitHub URL":
    github_url = st.text_input("Enter the raw GitHub PDF URL:")
    if github_url:
        try:
            response = requests.get(github_url)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(response.content)
                pdf_path = tmp_file.name
        except Exception as e:
            st.error(f"Error fetching PDF: {e}")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if pdf_path:
    st.success("PDF Loaded Successfully!")
    with st.spinner("Indexing PDF content..."):
        tmp_dir = tempfile.mkdtemp()
        text = extract_text_from_pdf(pdf_path)
        with open(os.path.join(tmp_dir, "document.txt"), "w", encoding="utf-8") as f:
            f.write(text)

        documents = SimpleDirectoryReader(tmp_dir).load_data()

        # ✅ Set local LLM and local embedding model
        Settings.llm = Ollama(model="llama2")
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        
    st.subheader("💬 Ask questions about your PDF")
    query = st.text_input("Your question:", placeholder="Ask something from the PDF...")
    if query:
        with st.spinner("Thinking..."):
            response = query_engine.query(query)
            st.markdown(
                f"""
                <div style='
                    background-color: #ffffff;
                    border: 1px solid #d3d3d3;
                    border-radius: 12px;
                    padding: 20px;
                    margin-top: 25px;
                    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
                '>
                    <div style='
                        font-family: "Segoe UI", sans-serif;
                        font-size: 15px;
                        color: #555;
                        margin-bottom: 10px;
                    '>
                        <strong>🧾 You asked:</strong><br>
                        <em>{query}</em>
                    </div>
                    <div style='
                        font-family: Georgia, serif;
                        font-style: italic;
                        font-size: 16px;
                        color: #333;
                        padding-top: 10px;
                        border-top: 1px dashed #ccc;
                    '>
                        <strong>🤖 Answer:</strong><br>
                        {response.response}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


else:
    st.info("Please upload a PDF or provide a valid GitHub URL to begin.")
