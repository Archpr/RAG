# Kerala University Admission Assistant

A RAG (Retrieval-Augmented Generation) based chatbot designed to assist students with admission-related queries for Kerala University entrace examinations

## 🚀 Key Features

*   **Hybrid Retrieval**: Combines ChromaDB (Vector Search) and Cross-Encoder Reranking (MS MARCO) for high-accuracy document retrieval.
*   **Intelligent Responses**: Uses Google Gemini (via LangChain) to generate helpful and context-aware answers.
*   **Dual Interfaces**:
    *   **Streamlit UI**: A user-friendly web interface with chat history, session management, and visual polish.
    *   **FastAPI Backend**: A robust REST API for integrating the chatbot into other applications.
*   **Source Citation**: Provides clear citations for the sources of information used in the answers.

## 🛠️ Tech Stack

*   **Language**: Python 3.8+
*   **LLM**: Google Gemini
*   **Orchestration**: LangChain, LangChain Community
*   **Vector Database**: ChromaDB
*   **Reranking**: Sentence Transformers (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
*   **Web Frameworks**: Streamlit (Frontend), FastAPI (Backend)
*   **PDF Processing**: PyPDF, PyMuPDF

##  Prerequisites

*   Python 3.8 or higher
*   Google Gemini API Key (set as an environment variable or in `.env` file)

##  Installation

1.  **Clone the repository**:
    ```bash
    git clone <https://github.com/Austin8547/RAG>
    cd <RAG>
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables**:
    Create a `.env` file in the root directory and add your Google API Key:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    ```

##  Usage

### Option 1: Streamlit UI
Run the interactive web application:
```bash
streamlit run streamlit_app.py
```
This will launch the app in your default web browser (usually at `http://localhost:8501`).

### Option 2: FastAPI Backend
Run the API server:
```bash
python app.py
```
Or using uvicorn directly:
```bash
uvicorn app:app --reload
```
The API will be available at `http://localhost:8000`. You can test the endpoints at `http://localhost:8000/docs`.
##  Project Structure

*   `main.py`: Main entry point for the application (can run both Streamlit and FastAPI).
*   `streamlit_app.py`: Main entry point for the Streamlit application.
*   `app.py`: Main entry point for the FastAPI application.
*   `test.py`: Contains unit and integration tests for the application.
*   `src/`: Source code directory.
    *   `ingestion/`: Logic for data ingestion and processing.
    *   `embeddings/`: Logic for generating and managing embeddings.
    *   `ragchain/`: Logic for the RAG pipeline.
    *   `retriever/`: Database and retrieval logic.
*   `static/`: Static assets (images, CSS, etc.).
*   `data/`: Source documents for the knowledge base.
