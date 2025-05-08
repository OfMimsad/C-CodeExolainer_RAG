import os
import json
import datetime
import logging
from multiprocessing import cpu_count
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from llama_cpp import Llama
import torch

from code_loader import load_cpp_files, chunk_code
from embedder import embed_chunks, save_faiss_index, load_faiss_index, load_metadata, get_embedding_model
from rag_query import search_index
from logger import setup_logger

# === Config ===
INDEX_FILE = "code_index.faiss"
META_FILE = "code_meta.json"
MEMORY_FILE = "chat_memory.json"
MAX_HISTORY = 5
MODEL_PATH = "models/mistral-7B-Instruct-v0.3/mistral.gguf"

setup_logger()
console = Console()

# === Hardware-aware model initialization ===
def get_safe_gpu_layers():
    if not torch.cuda.is_available():
        return 0
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 4.5:
            return 20
        elif vram_gb < 6.5:
            return 40
        else:
            return 100
    except Exception as e:
        logging.warning(f"⚠️ Could not determine GPU memory: {e}")
        return 0

def initialize_llm(model_path):
    gpu_layers = get_safe_gpu_layers()
    n_threads = min(cpu_count(), 12)
    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=gpu_layers,
            n_ctx=2048,
            n_threads=n_threads,
            verbose=False
        )
        logging.info(f"✅ Loaded model with {gpu_layers} GPU layers and {n_threads} threads.")
    except Exception as e:
        logging.warning(f"⚠️ GPU init failed. Falling back to CPU. Reason: {e}")
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=0,
            n_ctx=2048,
            n_threads=n_threads
        )
    return llm

llm = initialize_llm(MODEL_PATH)

# === Mistral Explain Wrapper ===
def explain_with_mistral(prompt):
    console.print("🤖 [bold blue]Asking Mistral for explanation...[/bold blue]")
    output = llm(
        prompt=f"[INST] {prompt} [/INST]",
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        stop=["</s>"]
    )
    if "choices" in output and output["choices"]:
        return output["choices"][0]["text"].strip()
    return "⚠️ Mistral returned no answer."

# === Indexing ===
def build_index(codebase_path):
    console.print("[bold magenta]🔧 Starting index build...[/bold magenta]")
    files = load_cpp_files(codebase_path)
    model, tokenizer = get_embedding_model()
    chunks = chunk_code(files, tokenizer, max_tokens=256)
    embeddings = embed_chunks(chunks, model, tokenizer)
    save_faiss_index(embeddings, chunks)
    console.print("[bold green]✅ Indexing complete.[/bold green]")


# === Querying ===
def query_index(user_query):
    try:
        index = load_faiss_index()
        metadata = load_metadata()
    except Exception as e:
        console.print(f"[bold red]❌ Failed to load index/metadata: {e}[/bold red]")
        return
    results = search_index(user_query, index, metadata)
    for i, r in enumerate(results, 1):
        console.print(f"\n[bold blue]Result {i}[/bold blue]")
        console.print(f"[bold]File:[/bold] {r['file']}")
        console.print(f"[italic]{r['content']}[/italic]")


# === Memory ===
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

def extract_tags(results):
    files = list({r["file"] for r in results})
    code_combined = " ".join(r["content"] for r in results)
    keywords = [w for w in code_combined.split() if len(w) > 4]
    common = max(set(keywords), key=keywords.count, default="code")
    return {"files": files, "topic": common.lower()}

def delete_last_codebase():
    for f in [INDEX_FILE, META_FILE]:
        if os.path.exists(f):
            os.remove(f)
            console.print(f"❌ Deleted [bold]{f}[/bold]")

def display_menu():
    table = Table(title="🛠️ Code Explainer Menu", show_lines=True)
    table.add_column("Option", style="cyan", justify="center")
    table.add_column("Action", style="magenta")
    table.add_row("1", "Ask a Question")
    table.add_row("2", "Delete Last Question")
    table.add_row("3", "Clear All Memory")
    table.add_row("4", "Upload New Codebase")
    table.add_row("5", "Exit")
    console.print(table)

# === MAIN LOOP ===
def main():
    try:
        chat_history = load_memory()
        console.print(f"🧠 Loaded [bold]{len(chat_history)}[/bold] previous memory entries.")

        index = None
        metadata = None
        if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
            index = load_faiss_index()
            metadata = load_metadata()

        while True:
            display_menu()
            choice = Prompt.ask("Choose an option").strip().lower()

            if choice in ["exit", "quit", "5"]:
                save_memory(chat_history)
                console.print("👋 Bye! Memory saved.", style="cyan")
                break

            elif choice == "1":
                if not index or not metadata:
                    console.print("⚠️ No codebase found. Please use option 4.", style="red")
                    continue
                query = Prompt.ask("💬 Ask about the code").strip()
                if query.lower() in ["back", "exit", ""]:
                    continue

                results = search_index(query, index, metadata)
                combined_code = "\n\n".join(r["content"] for r in results)

                history_prompt = "\n".join([f"User: {q['question']}\nAssistant: {q['answer']}" for q in chat_history[-MAX_HISTORY:]])
                prompt = f"{history_prompt}\nUser: {query}\nAssistant: Please explain the code:\n```\n{combined_code}\n```"

                answer = explain_with_mistral(prompt)
                console.print(Panel.fit(answer, title="💡 Mistral says", style="green"))

                chat_history.append({
                    "question": query,
                    "answer": answer,
                    "tags": extract_tags(results),
                    "timestamp": datetime.datetime.now().isoformat()
                })
                save_memory(chat_history)

            elif choice == "2":
                if chat_history:
                    confirm = Prompt.ask("Delete last question? (y/n)", default="n")
                    if confirm.lower() == "y":
                        chat_history.pop()
                        save_memory(chat_history)
                        console.print("🗑️ Last entry removed.", style="yellow")
                else:
                    console.print("⚠️ No memory to delete.", style="red")

            elif choice == "3":
                confirm = Prompt.ask("Clear all memory? (y/n)", default="n")
                if confirm.lower() == "y":
                    chat_history.clear()
                    save_memory(chat_history)
                    console.print("🧹 Memory cleared.", style="yellow")

            elif choice == "4":
                confirm = Prompt.ask("⚠️ This will erase current codebase. Continue? (y/n)", default="n")
                if confirm.lower() != "y":
                    continue
                delete_last_codebase()
                folder = Prompt.ask("📁 Enter codebase path").strip()
                if not os.path.exists(folder):
                    console.print("❌ Path does not exist.", style="red")
                    continue

                files = load_cpp_files(folder)
                model, tokenizer = get_embedding_model()
                chunks = chunk_code(files, tokenizer, max_tokens=256)
                
                try:
                    embedding_matrix, chunks = embed_chunks(chunks, model, tokenizer)
                    if embedding_matrix is not None and len(embedding_matrix) > 0:
                        save_faiss_index(embedding_matrix, chunks)
                        index = load_faiss_index()
                        metadata = load_metadata()
                        console.print("✅ Codebase indexed.", style="green")
                    else:
                         console.print("❌ Failed to generate embeddings.", style="red")
                except Exception as e:
                    console.print(f"🚨 Error during indexing: {e}", style="bold red")
                    logging.exception("Indexing error:")

            else:
                console.print("⚠️ Invalid choice.", style="red")

    except KeyboardInterrupt:
        console.print("\n👋 Exiting on interrupt.", style="cyan")
    except Exception as e:
        console.print(f"🚨 Fatal Error: {e}", style="bold red")
        logging.exception("Fatal error occurred:")

if __name__ == "__main__":
    main()