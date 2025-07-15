# Arabic RAG System

## Overview

This repository contains an Arabic Retrieval-Augmented Generation (RAG) system that allows you to query an Arabic text corpus using both classical search (BM25) and semantic search methods. The system compares the performance of both search methods and provides answers using a Gemini API.

## Files in this Repository

- `RAG_system.py`: The main application file to run
- `book.txt`: The source Arabic text document used for retrieval
- `embeddings.npz`: Pre-computed embeddings for the text chunks
- `index.faiss`: FAISS index file for faster vector search
- `chunks.pkl`: Processed text chunks used for retrieval

## Quick Start

1. **Ensure dependencies are installed:**
   ```
   pip install numpy faiss-cpu sentence-transformers rank-bm25 nltk dotenv
   ```

2. **Run the application:**
   ```
   python RAG_system.py
   ```

3. **That's it!** The GUI will open, allowing you to ask questions in Arabic.

## Features

- **Dual Search Methods**: Compare results from both classical search (BM25) and semantic vector search
- **Pre-computed Embeddings**: No need to create embeddings from scratch - everything is ready to go
- **LLM Integration**: Two types of answers are provided:
  - Answer without any retrieval
  - Answer using hybrid search retrieval
- **User-friendly Interface**: Simple GUI with RTL text support for Arabic

## Troubleshooting

If you encounter any issues:
1. Ensure all required files are in the same directory as RAG_system.py
2. Make sure all dependencies are properly installed