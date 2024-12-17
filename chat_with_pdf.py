import os
import pdfplumber
import gensim.downloader as api
import faiss
import numpy as np
from sklearn.preprocessing import normalize
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import scrolledtext

# Function to get all PDF files from a folder
def get_pdf_files_from_folder(folder_path):
    """Retrieve all PDF file paths from the specified folder."""
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".pdf")]

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_paths):
    """Extract text from a list of PDF files."""
    text_data = []
    for pdf_path in pdf_paths:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text_data.append(page.extract_text())
    return text_data

# Step 2: Chunking text into smaller segments
def chunk_text(text, chunk_size=300):
    """Split text into chunks of a specified size."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(' '.join(current_chunk)) >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

# Step 3: Generate vector embeddings for chunks
def generate_embeddings(chunks, model):
    """Generate vector embeddings for each chunk using a pre-trained model."""
    embeddings = []
    for chunk in chunks:
        words = chunk.split()
        valid_word_embeddings = [model[word] for word in words if word in model]
        if valid_word_embeddings:  # Check if there are valid word embeddings
            chunk_embedding = np.mean(valid_word_embeddings, axis=0)
            embeddings.append(chunk_embedding)
    return np.array(embeddings, dtype='float32')

# Step 4: Store embeddings in a FAISS vector database
def store_embeddings(embeddings):
    """Store embeddings in a FAISS index for efficient retrieval."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    normalized_embeddings = normalize(embeddings)
    index.add(normalized_embeddings)
    return index

# Step 5: Retrieve relevant chunks for a query
def retrieve_relevant_chunks(query, index, model, chunks, top_k=3):
    """Retrieve the most relevant chunks based on the query."""
    valid_query_embeddings = [model[word] for word in query.split() if word in model]
    if not valid_query_embeddings:  # If no valid words in the query
        return ["No relevant information found for the given query."]

    query_embedding = np.mean(valid_query_embeddings, axis=0).reshape(1, -1)
    query_embedding = normalize(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# Main function for RAG pipeline
def rag_pipeline(pdf_paths, query):
    """Run the RAG pipeline on given PDFs and query."""
    # Load embedding model
    model = api.load("glove-wiki-gigaword-100")

    # Extract and process text
    extracted_text = extract_text_from_pdf(pdf_paths)
    all_chunks = []
    for page_text in extracted_text:
        all_chunks.extend(chunk_text(page_text))

    # Generate embeddings and store in vector database
    chunk_embeddings = generate_embeddings(all_chunks, model)
    vector_index = store_embeddings(chunk_embeddings)

    # Retrieve relevant chunks for the query
    relevant_chunks = retrieve_relevant_chunks(query, vector_index, model, all_chunks)

    # Generate response
    response = "\n".join(relevant_chunks[:3])  # Limit to top 3 relevant chunks
    return response

# GUI Application
def create_gui():
    def browse_folder():
        folder = filedialog.askdirectory()
        folder_path_var.set(folder)

    def process_query():
        folder_path = folder_path_var.get()
        query = query_var.get()

        if not folder_path or not query:
            messagebox.showerror("Error", "Please select a folder and enter a query.")
            return

        pdf_files = get_pdf_files_from_folder(folder_path)
        if not pdf_files:
            messagebox.showerror("Error", "No PDF files found in the selected folder.")
            return

        try:
            result = rag_pipeline(pdf_files, query)
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    # Create the main window
    root = tk.Tk()
    root.title("CHAT WITH PDF USING RAG PIPELINE")

    # Folder selection
    folder_frame = tk.Frame(root)
    folder_frame.pack(pady=10)

    folder_label = tk.Label(folder_frame, text="Select Folder:")
    folder_label.pack(side=tk.LEFT, padx=5)

    folder_path_var = tk.StringVar()
    folder_entry = tk.Entry(folder_frame, textvariable=folder_path_var, width=50)
    folder_entry.pack(side=tk.LEFT, padx=5)

    browse_button = tk.Button(folder_frame, text="Browse", command=browse_folder)
    browse_button.pack(side=tk.LEFT, padx=5)

    # Query input
    query_frame = tk.Frame(root)
    query_frame.pack(pady=10)

    query_label = tk.Label(query_frame, text="Enter Query:")
    query_label.pack(side=tk.LEFT, padx=5)

    query_var = tk.StringVar()
    query_entry = tk.Entry(query_frame, textvariable=query_var, width=50)
    query_entry.pack(side=tk.LEFT, padx=5)

    # Process button
    process_button = tk.Button(root, text="Process", command=process_query)
    process_button.pack(pady=10)

    # Result display
    result_text_frame = tk.Frame(root)
    result_text_frame.pack(pady=10)

    result_text = scrolledtext.ScrolledText(result_text_frame, height=15, width=80, wrap=tk.WORD)
    result_text.pack(fill=tk.BOTH, expand=True)

    # Run the GUI loop
    root.mainloop()

if __name__ == "__main__":
    create_gui()
