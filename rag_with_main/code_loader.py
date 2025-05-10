import os
from concurrent.futures import ThreadPoolExecutor
from rich.console import Console

console = Console()

def read_file(path):
    """Read a file and return its absolute path and content."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if not content.strip():
                console.print(f"[bold yellow]⚠️ File {path} is empty or contains only whitespace[/bold yellow]")
                return os.path.abspath(path), None
            return os.path.abspath(path), content
    except Exception as e:
        console.print(f"[bold red]❌ Error reading file {path}: {e}[/bold red]")
        return os.path.abspath(path), None

def load_cpp_files(folder):
    """Load all C++ files from the specified folder."""
    cpp_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith((".cpp", ".h", ".hpp", ".cc", ".cxx", ".c")):
                cpp_paths.append(os.path.join(root, file))

    console.print(f"[bold cyan]📄 Found {len(cpp_paths)} C++ files to load[/bold cyan]")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        cpp_files = list(executor.map(read_file, cpp_paths))
    
    # Filter out files with None content
    valid_files = [(path, content) for path, content in cpp_files if content is not None]
    console.print(f"[bold green]✅ Loaded {len(valid_files)} valid C++ files[/bold green]")
    return valid_files

def chunk_code(files, tokenizer, max_tokens=200, stride=100):
    """Token-based chunking: handles tuples like (filename, content)."""
    chunks = []
    total_chunks = 0
    console.print("[bold magenta]📦 Chunking code files...[/bold magenta]")

    for filename, code in files:
        try:
            tokens = tokenizer(code, return_tensors="pt", truncation=False)["input_ids"][0]
            console.print(f"[bold blue]🔍 File {filename}: {len(tokens)} tokens[/bold blue]")
            num_chunks = 0

            if len(tokens) < max_tokens:
                # Handle short files by creating a single chunk
                chunk_text = tokenizer.decode(tokens, skip_special_tokens=True)
                if chunk_text.strip():
                    chunks.append({
                        "file": filename,
                        "chunk": chunk_text.strip()
                    })
                    num_chunks += 1
            else:
                # Chunk longer files
                for i in range(0, len(tokens) - max_tokens + 1, stride):
                    chunk_tokens = tokens[i:i + max_tokens]
                    chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    if chunk_text.strip():
                        chunks.append({
                            "file": filename,
                            "chunk": chunk_text.strip()
                        })
                        num_chunks += 1

            total_chunks += num_chunks
            console.print(f"[bold green]✅ {filename}: {num_chunks} chunks[/bold green]")

        except Exception as e:
            console.print(f"[bold red]❌ Error chunking file {filename}: {e}[/bold red]")

    console.print(f"[bold green]✅ Total chunks created: {total_chunks}[/bold green]")
    return chunks