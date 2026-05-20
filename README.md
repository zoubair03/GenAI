# Embedded Systems AI Assistant (RAG Pipeline)

This project is a specialized AI Chatbot designed to assist Embedded Systems Engineers. It acts as a "Senior Hardware Architect," capable of searching through technical datasheets, calculating hardware formulas (like Ohm's Law), and writing microcontroller firmware (MicroPython/C++).

The core architecture relies on a **RAG (Retrieval-Augmented Generation)** system paired with a **ReAct Agent** powered by local LLMs.

---

## 🧠 What is RAG (Retrieval-Augmented Generation)?

Large Language Models (LLMs) like Llama 3 or ChatGPT are trained on vast amounts of data, but they have two major flaws:
1. **They hallucinate (guess) facts.**
2. **They don't know your private data.** (e.g., they don't have access to specific, niche manufacturer datasheets you downloaded today).

**RAG solves this by providing the AI with an "open-book test."** 
Instead of relying on the LLM's memory to answer a question, a RAG system first **Retrieves** the exact relevant documents from a database, **Augments** (adds) that text into the prompt, and asks the LLM to **Generate** an answer based *only* on the provided text.

### How the Pipeline Works in This Project:
1. **Ingestion**: You put PDF datasheets into the `iot_devices_datasheets` folder.
2. **Chunking**: The app reads the PDFs and chops the text into smaller, overlapping chunks.
3. **Embedding**: A math model converts these text chunks into numerical vectors (lists of numbers) that capture the "meaning" of the text.
4. **Vector Database**: These vectors are saved into ChromaDB.
5. **Retrieval**: When a user asks "What is the ESP32 voltage?", the system turns the question into a vector, searches ChromaDB for the closest matching text chunks, and pulls out the exact paragraphs from the datasheet.
6. **Generation**: The LLM reads those specific paragraphs and formulates a final, human-readable answer.

---

## 🛠️ Libraries Used & Their Purposes

This project uses the modern AI stack, primarily orchestrated by **LangChain**.

### 1. `langchain` / `langchain_classic`
LangChain is the orchestrator. It provides the glue that connects the PDF loaders, the vector database, the tools, and the LLM together.
* **AgentExecutor & create_react_agent**: This sets up the "brain" of the chatbot. A ReAct (Reasoning + Acting) agent doesn't just answer questions; it thinks ("I need to search the datasheet"), takes an action ("Using Datasheet_Search tool"), observes the result, and then decides what to do next.

### 2. `langchain_community.document_loaders` (`PyPDFLoader`)
Used to parse raw `.pdf` files from your local directory and extract the raw text so the AI can process it.

### 3. `langchain_text_splitters` (`RecursiveCharacterTextSplitter`)
You cannot feed an entire 100-page PDF into an LLM at once (it exceeds the context window). This library chops the PDF text into chunks of 1000 characters, keeping a 150-character overlap so that sentences aren't awkwardly cut in half.

### 4. `langchain_huggingface` (`HuggingFaceEmbeddings`)
This is the mathematical engine for the RAG search. It uses the `all-MiniLM-L6-v2` model from Hugging Face to convert English text into vector numbers. It runs locally on your CPU/GPU, ensuring your private datasheets are never sent to the cloud.

### 5. `langchain_chroma` (`Chroma`)
ChromaDB is the Vector Database. Think of it like a specialized SQL database, but instead of searching for exact text matches, it searches for "concepts" using the vector numbers created by Hugging Face.

### 6. `langchain_ollama` (`ChatOllama`)
Ollama is a tool that allows you to run massive open-source LLMs (like Meta's `llama3`) entirely locally on your own hardware. `ChatOllama` is the LangChain wrapper that talks to your local Ollama server to generate the final text responses and code.

### 7. `numexpr`
A fast numerical expression evaluator for Python. Instead of asking the LLM to "do math" (which LLMs are notoriously bad at), we gave the Agent a "Calculator" tool. The LLM creates the formula (e.g., `3.3 / 330`), and `numexpr` safely and instantly calculates the exact result.

### 8. `streamlit`
The web framework. Streamlit allows you to build beautiful, interactive chat interfaces and dashboards using pure Python, without needing to know HTML, CSS, or JavaScript.

---

## 🚀 How to Run
1. Ensure your Python virtual environment (`.venv`) is activated.
2. Ensure you have installed all dependencies (including `torchvision` for the embeddings).
3. Ensure the Ollama application is running in the background.
4. Run the app:
   ```bash
   streamlit run app.py
   ```