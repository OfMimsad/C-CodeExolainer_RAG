import os
from concurrent.futures import ThreadPoolExecutor

def read_file(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            # Use full absolute path instead of relpath to avoid cross-drive issues
            return os.path.abspath(path), f.read()
    except Exception as e:
        return os.path.abspath(path), f"// Error reading file: {e}"

def load_cpp_files(folder):
    cpp_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith((".cpp", ".h", ".hpp", ".cc", ".cxx", ".c")):
                cpp_paths.append(os.path.join(root, file))

    with ThreadPoolExecutor(max_workers=8) as executor:
        cpp_files = list(executor.map(read_file, cpp_paths))

    return cpp_files

def chunk_code(files, tokenizer, max_tokens=256, stride=128):
    """
    Token-based chunking: handles tuples like (filename, content)
    """
    chunks = []
    for filename, code in files:
        tokens = tokenizer(code, return_tensors="pt", truncation=False)["input_ids"][0]
        for i in range(0, len(tokens) - max_tokens + 1, stride):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append({
                "file": filename,
                "chunk": chunk_text.strip()  # Changed from "content" to "chunk"
            })
    return chunks
