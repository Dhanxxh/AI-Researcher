import gradio as gr
import os
import time
from dotenv import load_dotenv

# AI & RAG
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
# OCR
import pytesseract
from PIL import Image

# Set Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

load_dotenv()

app_agent = None

def process_file(file):
    global app_agent
    groq_key = os.getenv("GROQ_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not groq_key or not tavily_key:
        return "❌ Error: API keys missing in .env (GROQ_API_KEY and TAVILY_API_KEY)"

    try:
        file_path = file.name
        file_ext = os.path.splitext(file_path)[1].lower()

        # --- Load documents based on file type ---
        if file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()

        elif file_ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
            if not text.strip():
                return "❌ No text found in image. Make sure the image contains readable text."
            docs = [Document(page_content=text, metadata={"source": file_path})]

        else:
            return f"❌ Unsupported file type: {file_ext}. Please upload a PDF or image."

        # --- Split ---
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # --- Embeddings ---
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # --- FAISS Vector Store ---
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()

        # --- Tools ---
        rag_tool = create_retriever_tool(
            retriever,
            "document_search",
            "Search for info inside the uploaded document or image. Check this first for document questions."
        )
        search_tool = TavilySearchResults(max_results=2, tavily_api_key=tavily_key)

        # --- LLM ---
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=groq_key,
            temperature=0
        )

        # --- Agent ---
        app_agent = create_react_agent(llm, [rag_tool, search_tool])

        file_type = "Image (OCR)" if file_ext != ".pdf" else "PDF"
        return f"✅ Agent Ready! Loaded: {os.path.basename(file_path)} ({file_type})"

    except Exception as e:
        return f"❌ Setup Error: {str(e)}"


def chat(message, history):
    global app_agent
    if app_agent is None:
        return "Upload a PDF or Image first."

    try:
        inputs = {"messages": [("human", message)]}
        result = app_agent.invoke(inputs)
        return result["messages"][-1].content
    except Exception as e:
        return f"⚠️ Chat Error: {str(e)}"


# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Professional Agentic Researcher")
    gr.Markdown("Local Data Processing + **Llama 3 (Groq)** Reasoning + Web Search")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload PDF or Image",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]
            )
            status = gr.Textbox(label="Status", interactive=False)
            file_input.change(fn=process_file, inputs=file_input, outputs=status)
        with gr.Column(scale=2):
            gr.ChatInterface(fn=chat, title="Chat with Research Agent")

if __name__ == "__main__":
    demo.launch(share=True)