import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from rich.console import Console

console = Console()

# === Safe device selection with fallback ===
def get_device():
    try:
        if torch.cuda.is_available():
            console.print("[bold green]✅ CUDA is available. Using GPU for RAG query.[/bold green]")
            return torch.device("cuda")
        else:
            raise RuntimeError("CUDA not available")
    except Exception as e:
        console.print(f"[bold yellow]⚠️ Falling back to CPU: {e}[/bold yellow]")
        return torch.device("cpu")

device = get_device()

# === Load the same model used in embedder.py ===
try:
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(device)
    console.print(f"[bold green]📚 BGE model loaded on {device}[/bold green]")
except Exception as e:
    console.print(f"[bold red]❌ Failed to load BGE model: {e}[/bold red]")
    raise

# === Embed query with same method as embedder.py ===
def embed_query(text):
    try:
        encoded_input = tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = model(**encoded_input).last_hidden_state[:, 0]  # CLS token
        embedding = embedding.cpu().numpy()
        return embedding
    except Exception as e:
        console.print(f"[bold red]🚨 Query embedding error: {str(e)}[/bold red]")
        return np.array([])

# === Search FAISS index ===
def search_index(query, index, metadata, top_k=5):
    embedding = embed_query(query)
    if embedding.size == 0:
        return []

    try:
        distances, indices = index.search(embedding, top_k)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            console.print("[bold blue]🚿 Cleaned up GPU memory after search[/bold blue]")

        results = []
        for idx in indices[0]:
            if idx < len(metadata):
                results.append({
                    "file": metadata[idx]["file"],
                    "content": metadata[idx]["chunk_preview"]
                })

        console.print(f"[bold yellow]🔎 Retrieved {len(results)} results for query:[/bold yellow] [italic]{query[:40]}...[/italic]")
        return results

    except Exception as e:
        console.print(f"[bold red]🚨 Error during FAISS search: {str(e)}[/bold red]")
        return []
