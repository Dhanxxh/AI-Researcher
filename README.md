# 🤖 Agentic RAG Researcher

An advanced **AI-powered research assistant** that analyzes PDFs and images, extracts text using OCR, and answers questions using **Retrieval-Augmented Generation (RAG)** with real-time web search.

---

## 🚀 Features

* 📄 Upload and analyze **PDF documents**
* 🖼️ Extract text from **images using OCR (Tesseract)**
* 🔍 Semantic search using **FAISS vector database**
* 🌐 Real-time web search via **Tavily API**
* 🧠 AI reasoning with **Llama 3 (Groq API)**
* 💬 Interactive chat UI using **Gradio**
* ⚡ Fast embeddings with **HuggingFace**

---

## 🧠 How It Works

1. Upload a PDF or image
2. Extract text (OCR for images)
3. Split text into chunks
4. Convert text into embeddings
5. Store in FAISS vector database
6. Agent decides:

   * Retrieve from document (RAG)
   * OR search the web
7. LLM generates the final answer

---

## 🛠️ Tech Stack

* **Python**
* **LangChain + LangGraph**
* **Groq (Llama 3)**
* **HuggingFace Embeddings**
* **FAISS**
* **Gradio**
* **Tesseract OCR**
* **Tavily Search API**

---

## 📂 Project Structure

```
├── app.py
├── simple.py
├── list_models.py
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/agentic-rag-researcher.git
cd agentic-rag-researcher
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Install Tesseract OCR

Download: https://github.com/tesseract-ocr/tesseract

Then update path in `app.py` if needed:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

---

## 🔑 Environment Variables

Create a `.env` file in root:

```
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_API_KEY=your_google_api_key
```

---

## ▶️ Run the Application

```bash
python app.py
```

Then open the Gradio link in your browser.

---

## 💬 Usage

1. Upload a PDF or image
2. Wait for "Agent Ready"
3. Ask questions
4. Get answers from document + web

---

## 📌 Use Cases

* Research paper analysis
* Resume/document Q&A
* Image text extraction
* Hybrid AI search (document + internet)

---

## ⚠️ Notes

* Do NOT upload `.env` or `venv/` to GitHub
* Tesseract installation is required for OCR
* Internet required for APIs

---

## 🚀 Future Improvements

* Multi-document support
* Chat memory
* Better UI
* Cloud deployment

---

## 🤝 Contributing

Feel free to fork and improve this project.

---

## 📜 License

MIT License
