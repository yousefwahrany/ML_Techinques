import os
import re
import sys
import threading
import pickle
import numpy as np
import faiss
import tkinter as tk
from tkinter import scrolledtext
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

# Set UTF-8 for stdout
sys.stdout.reconfigure(encoding='utf-8')

# Download Arabic stopwords
nltk.download('stopwords', quiet=True)
arabic_stopwords = set(stopwords.words('arabic'))

# ----------------------------
# STEP 0: Cleaning Functions
# ----------------------------

def clean_arabic_text_gui(text):
    # For general GUI cleaning
    pattern = r"[^؀-ۿݐ-ݿࢠ-ࣿ0-9a-zA-Z٠-٩\s.,!?؟،؛\-\n]"
    return re.sub(pattern, '', text)

def normalize_arabic(text):
    # For classical search normalization
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^\w\s.]', '', text)  # Keep fullstops
    return text

def clean_arabic_text_semantic(text):
    # For semantic search normalization
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    return text

def simple_tokenize(text):
    return text.split()

def preprocess(text):
    # For BM25 tokenization
    text = normalize_arabic(text)
    tokens = simple_tokenize(text)
    tokens = [t for t in tokens if t not in arabic_stopwords and len(t) > 1]
    return tokens

# ----------------------------
# STEP 1: Load & Chunk Text
# ----------------------------

def load_and_chunk_text(file_path, sentences_per_chunk=5, cleaned_output_path="cleaned_output.txt"):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Clean the text
    text = clean_arabic_text_gui(text)

    # Save cleaned text for reference
    with open(cleaned_output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    # Split text into sentences
    sentences = re.split(r'(?<=[.؟!])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    # Create overlapping chunks
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)

    return chunks, sentences

# ----------------------------
# STEP 2: Classical Search with BM25
# ----------------------------

def setup_classical_search(chunks):
    # Preprocess documents for BM25
    tokenized_corpus = [preprocess(doc) for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def classical_search(query, bm25, chunks, top_k=5):
    # Preprocess the query
    tokenized_query = preprocess(query)
    # Get BM25 scores
    scores = bm25.get_scores(tokenized_query)
    # Get top documents
    top_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    for doc_idx, score in top_docs:
        results.append((chunks[doc_idx], score))
    
    return results

# ----------------------------
# STEP 3: Semantic Search with FAISS
# ----------------------------

def generate_embeddings(text_chunks, model):
    print("Generating embeddings...")
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    return embeddings

def build_faiss_index(embeddings):
    # Build FAISS index for cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def semantic_search(query, model, index, chunks, top_k=5):
    # Encode query
    query_embedding = model.encode([query]).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # Search index
    D, I = index.search(query_embedding, top_k)
    
    # Format results
    results = []
    for idx, i in enumerate(I[0]):
        if idx < len(D[0]):  # Ensure we don't go out of bounds
            results.append((chunks[i], D[0][idx]))
    
    return results

# ----------------------------
# STEP 4: Hybrid Retrieval
# ----------------------------

def hybrid_retrieve(query, model, faiss_index, documents, top_k_semantic=20, top_k_final=10):
    query_embedding = model.encode([query]).astype('float32')
    faiss.normalize_L2(query_embedding)
    D, I = faiss_index.search(query_embedding, top_k_semantic)

    candidate_chunks = [documents[i] for i in I[0]]
    tokenized_query = preprocess(query)
    candidate_tokens = [preprocess(doc) for doc in candidate_chunks]

    bm25 = BM25Okapi(candidate_tokens)
    scores = bm25.get_scores(tokenized_query)
    top_reranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k_final]

    top_chunks = [candidate_chunks[idx] for idx, _ in top_reranked]
    return top_chunks

# ----------------------------
# STEP 5: Save/Load Functions
# ----------------------------

def save_embeddings_and_index(embeddings, index, chunks, bm25_model, 
                             index_file='index.faiss', 
                             embed_file='embeddings.npz', 
                             chunk_file='chunks.pkl',
                             bm25_file='bm25.pkl'):
    # Save FAISS index
    faiss.write_index(index, index_file)
    
    # Save embeddings
    np.savez(embed_file, embeddings=embeddings)
    
    # Save chunks
    with open(chunk_file, 'wb') as f:
        pickle.dump(chunks, f)
    
    # Save BM25 model
    with open(bm25_file, 'wb') as f:
        pickle.dump(bm25_model, f)

def load_embeddings_and_index(index_file='index.faiss', 
                             embed_file='embeddings.npz', 
                             chunk_file='chunks.pkl',
                             bm25_file='bm25.pkl'):
    # Load FAISS index
    index = faiss.read_index(index_file)
    
    # Load embeddings
    embeddings = np.load(embed_file)['embeddings']
    
    # Load chunks
    with open(chunk_file, 'rb') as f:
        chunks = pickle.load(f)
    
    # Load BM25 model
    with open(bm25_file, 'rb') as f:
        bm25_model = pickle.load(f)
    
    return embeddings, index, chunks, bm25_model

# ============================
# GEMINI LLM INTEGRATION
# ============================

def setup_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

def get_gemini_response(model, prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting response: {str(e)}"

# ----------------------------
# STEP 6: GUI Components
# ----------------------------

def format_search_results(results, search_type):
    formatted = f"\n--- {search_type} Search Results ---\n"
    for i, (chunk, score) in enumerate(results, start=1):
        formatted += f"{i}. Score: {score:.4f}\n{chunk}\n\n"
    return formatted

def send_prompt(input_box, output_widget, gemini_model, model, faiss_index, chunks, bm25_model):
    prompt = input_box.get()
    if not prompt.strip():
        return
    input_box.delete(0, tk.END)
    
    # Create a new thread to handle the processing
    threading.Thread(
        target=process_query, 
        args=(prompt, output_widget, gemini_model, model, faiss_index, chunks, bm25_model)
    ).start()

def process_query(prompt, output_widget, gemini_model, model, faiss_index, chunks, bm25_model):
    output_widget.config(state='normal')
    output_widget.insert(tk.END, f"\n--- NEW QUERY ---\n")
    output_widget.insert(tk.END, f"You: {prompt}\n\n")
    
    # Get classical search results
    classical_results = classical_search(prompt, bm25_model, chunks)
    output_widget.insert(tk.END, format_search_results(classical_results, "Classical (BM25)"))
    
    # Get semantic search results
    semantic_results = semantic_search(prompt, model, faiss_index, chunks)
    output_widget.insert(tk.END, format_search_results(semantic_results, "Semantic"))
    
    # Get hybrid results
    hybrid_results = hybrid_retrieve(prompt, model, faiss_index, chunks)
    
    # Get direct Gemini response (without retrieval)
    output_widget.insert(tk.END, "Gemini 1.5 Flash (without retrieval):\n")
    direct_response = get_gemini_response(gemini_model, prompt)
    output_widget.insert(tk.END, f"{direct_response}\n\n")
    
    # Get Gemini response with classical search context
    # classical_context = "\n".join([chunk for chunk, _ in classical_results])
    # classical_prompt = f"({prompt}) استخدم هذه الفقرات كسياق للإجابة على هذا السؤال ولا تذكر أنك تجيب باستخدام فقرات وإنما عليك التظاهر بأنك تعلم هذه الأشياء بالفعل: {classical_context}"
    
    # output_widget.insert(tk.END, "Gemini 1.5 Flash (with classical retrieval):\n")
    # classical_response = get_gemini_response(gemini_model, classical_prompt)
    # output_widget.insert(tk.END, f"{classical_response}\n\n")
    
    # Get Gemini response with hybrid retrieval context
    hybrid_context = "\n".join(hybrid_results)
    hybrid_prompt = f"({prompt}) استخدم هذه الفقرات كسياق للإجابة على هذا السؤال ولا تذكر أنك تجيب باستخدام فقرات وإنما عليك التظاهر بأنك تعلم هذه الأشياء بالفعل: {hybrid_context}"
    
    output_widget.insert(tk.END, "Gemini 1.5 Flash (with hybrid retrieval):\n")
    hybrid_response = get_gemini_response(gemini_model, hybrid_prompt)
    output_widget.insert(tk.END, f"{hybrid_response}\n\n")
    
    output_widget.insert(tk.END, "=" * 80 + "\n")
    output_widget.see(tk.END)  # Scroll to the bottom
    output_widget.config(state='disabled')

def main():
    # File and configuration paths
    file_path = "book.txt"
    index_file = 'index.faiss'
    embed_file = 'embeddings.npz'
    chunk_file = 'chunks.pkl'
    bm25_file = 'bm25.pkl'
    
    # Load or create embeddings and indexes
    if os.path.exists(index_file) and os.path.exists(embed_file) and os.path.exists(chunk_file) and os.path.exists(bm25_file):
        print("Loading saved embeddings and index...")
        embeddings, faiss_index, chunks, bm25_model = load_embeddings_and_index(
            index_file, embed_file, chunk_file, bm25_file
        )
        print("Loaded saved data successfully!")
    else:
        print("Processing text and generating new indexes...")
        # Load and chunk text
        chunks, sentences = load_and_chunk_text(file_path)
        
        # Load embedding model
        model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
        print("Embedding model loaded.")
        
        # Generate embeddings
        embeddings = generate_embeddings(chunks, model)
        print("Embeddings generated.")
        
        # Build FAISS index
        faiss_index = build_faiss_index(embeddings)
        print("FAISS index built.")
        
        # Setup BM25 model
        bm25_model = setup_classical_search(chunks)
        print("BM25 model created.")
        
        # Save everything for future use
        save_embeddings_and_index(embeddings, faiss_index, chunks, bm25_model,
                                 index_file, embed_file, chunk_file, bm25_file)
        print("All data saved for future use.")
    
    # Set up Gemini
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = setup_gemini(gemini_api_key)
    print("Gemini model initialized.")
    
    # The embedding model for query encoding
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
    print("Embedding model loaded for queries.")
    
    # Set up the GUI
    root = tk.Tk()
    root.title("Arabic Text Retrieval System")
    
    output = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=30, state='disabled')
    output.pack(padx=10, pady=10)
    
    input_frame = tk.Frame(root)
    input_frame.pack(pady=5)
    
    input_box = tk.Entry(input_frame, width=80)
    input_box.pack(side=tk.LEFT, padx=(10, 5))
    input_box.bind("<Return>", lambda event: send_prompt(
        input_box, output, gemini_model, model, faiss_index, chunks, bm25_model
    ))
    
    send_button = tk.Button(
        input_frame, 
        text="Send", 
        command=lambda: send_prompt(input_box, output, gemini_model, model, faiss_index, chunks, bm25_model)
    )
    send_button.pack(side=tk.LEFT)
    
    # Add welcome message
    output.config(state='normal')
    output.insert(tk.END, "Welcome to the Arabic Text Retrieval System!\n")
    output.insert(tk.END, "Enter your query in the box below and press Send or Enter.\n")
    output.insert(tk.END, "=" * 80 + "\n")
    output.config(state='disabled')
    
    root.mainloop()

if __name__ == "__main__":
    main()