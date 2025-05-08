import os
import json
import faiss  # type: ignore
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from rich.console import Console
from tqdm import tqdm

# === Setup ===
console = Console()
INDEX_FILE = "code_index.faiss"
META_FILE = "code_meta.json"

# === Safe device selection ===
def get_device():
    try:
        if torch.cuda.is_available():
            console.print("[bold green]✅ CUDA available. Using GPU![/bold green]")
            return torch.device("cuda")
        else:
            raise RuntimeError("CUDA not available")
    except Exception as e:
        console.print(f"[bold yellow]⚠️ Falling back to CPU: {e}[/bold yellow]")
        return torch.device("cpu")

device = get_device()

# === Load embedding model and tokenizer ===
try:
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(device)
    console.print("[bold green]📚 Embedding model loaded successfully![/bold green]")
except Exception as e:
    console.print(f"[bold red]❌ Failed to load embedding model: {e}[/bold red]")
    raise

# === File loader ===
def load_cpp_files(folder_path):
    files = []
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(('.cpp', '.hpp', '.h', '.c')):
                full_path = os.path.join(root, filename)
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    files.append((os.path.relpath(full_path, folder_path), f.read()))
    console.print(f"[bold cyan]📄 Loaded {len(files)} C/C++ files from {folder_path}[/bold cyan]")
    return files

# === Chunk code ===
def chunk_code(files, tokenizer, max_tokens=256, stride=128):
    chunks = []
    total_chunks = 0
    print("📦 Chunking code files...")

    for filename, code in tqdm(files):
        tokens = tokenizer(code, return_tensors="pt", truncation=False)["input_ids"][0]
        num_chunks = 0

        for i in range(0, len(tokens) - max_tokens + 1, stride):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            chunks.append({
                "file": filename,
                "chunk": chunk_text.strip()
            })
            num_chunks += 1

        total_chunks += num_chunks
        print(f"✅ {filename}: {num_chunks} chunks")

    print(f"\n✅ Total chunks created: {total_chunks}\n")
    return chunks

# === Embed chunks ===
def embed_chunks(chunks, model, tokenizer, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    embeddings = []
    metadata = []

    print(f"🚀 Embedding {len(chunks)} chunks using {device.upper()}")

    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        texts = [chunk["chunk"] for chunk in batch]

        try:
            encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                model_output = model(**encoded_input)

            emb_batch = model_output.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(emb_batch)

            metadata.extend(batch)  # Store original 'file' and 'chunk' for each

        except Exception as e:
            print(f"❌ Error embedding batch {i}-{i+batch_size}: {e}")

    all_embeddings = np.vstack(embeddings)
    print(f"\n✅ Total embedded chunks: {len(metadata)}\n")
    return all_embeddings, metadata

# === Save FAISS index ===
def save_faiss_index(embedding_matrix, metadata, index_file=INDEX_FILE, meta_file=META_FILE):
    if embedding_matrix.shape[0] == 0:
        console.print("[bold red]⚠️ No embeddings to save. Skipping FAISS index save.[/bold red]")
        return

    dim = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embedding_matrix)

    faiss.write_index(index, index_file)

    preview_metadata = [
        {"file": entry["file"], "chunk_preview": entry["chunk"][:300]}
        for entry in metadata
    ]
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(preview_metadata, f, indent=2)

    console.print(f"[bold green]💾 Saved FAISS index to {index_file} and metadata to {meta_file}[/bold green]")

# === Loaders ===
def load_faiss_index(index_file=INDEX_FILE):
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"FAISS index file '{index_file}' not found.")
    index = faiss.read_index(index_file)
    console.print(f"[bold cyan]📥 Loaded FAISS index from {index_file}[/bold cyan]")
    return index

def load_metadata(meta_file=META_FILE):
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"Metadata file '{meta_file}' not found.")
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    console.print(f"[bold cyan]📖 Loaded metadata from {meta_file}[/bold cyan]")
    return metadata

# === Getter ===
def get_embedding_model():
    return model, tokenizer
