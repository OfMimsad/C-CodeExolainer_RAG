import os
import json
from tree_sitter import Language, Parser
import tree_sitter_cpp
from rich.console import Console
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

console = Console()

# Initialize tree-sitter parser for C++
CPP_LANGUAGE = Language(tree_sitter_cpp.language())
parser = Parser()

# Try setting language with modern API, fallback to older API if needed
try:
    parser.set_language(CPP_LANGUAGE)
except AttributeError:
    console.print("[bold yellow]⚠️ Using older tree-sitter API for language setting[/bold yellow]")
    parser.language = CPP_LANGUAGE  # Fallback for older versions

# Symbol table structure
symbol_table = {
    "functions": {},  # function_name -> {file, declaration, calls}
    "classes": {},    # class_name -> {file, declaration}
    "variables": {}   # variable_name -> {file, declaration}
}

def parse_file(file_path):
    """Parse a single C++ file and extract symbols and their relationships."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        tree = parser.parse(code.encode("utf-8"))
        root_node = tree.root_node
        file_symbols = {
            "file": os.path.abspath(file_path),
            "functions": [],
            "calls": [],
            "classes": [],
            "variables": []
        }

        def traverse(node, depth=0):
            """Recursively traverse AST to find symbols."""
            if node.type in ["function_definition", "function_declarator"]:
                # Extract function name
                for child in node.children:
                    if child.type == "identifier":
                        func_name = child.text.decode("utf-8")
                        file_symbols["functions"].append({
                            "name": func_name,
                            "line": node.start_point[0] + 1
                        })
                        break
            elif node.type == "class_specifier":
                # Extract class name
                for child in node.children:
                    if child.type == "identifier":
                        class_name = child.text.decode("utf-8")
                        file_symbols["classes"].append({
                            "name": class_name,
                            "line": node.start_point[0] + 1
                        })
                        break
            elif node.type == "declaration" and "identifier" in [c.type for c in node.children]:
                # Extract variable name
                for child in node.children:
                    if child.type == "identifier":
                        var_name = child.text.decode("utf-8")
                        file_symbols["variables"].append({
                            "name": var_name,
                            "line": node.start_point[0] + 1
                        })
                        break
            elif node.type == "call_expression":
                # Extract function calls
                for child in node.children:
                    if child.type == "identifier":
                        call_name = child.text.decode("utf-8")
                        file_symbols["calls"].append({
                            "name": call_name,
                            "line": node.start_point[0] + 1
                        })
                        break

            for child in node.children:
                traverse(child, depth + 1)

        traverse(root_node)
        return file_path, file_symbols
    except Exception as e:
        console.print(f"[bold yellow]⚠️ Error parsing {file_path}: {e}[/bold yellow]")
        return file_path, None

def build_symbol_table(codebase_path):
    """Build symbol table for the entire codebase."""
    global symbol_table
    symbol_table = {"functions": {}, "classes": {}, "variables": {}}
    cpp_paths = []

    # Collect C++ files
    for root, _, files in os.walk(codebase_path):
        for file in files:
            if file.endswith((".cpp", ".h", ".hpp", ".cc", ".cxx", ".c")):
                cpp_paths.append(os.path.join(root, file))

    console.print(f"[bold cyan]📄 Found {len(cpp_paths)} C++ files to analyze[/bold cyan]")

    # Parallel parsing
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(parse_file, cpp_paths), total=len(cpp_paths), desc="Parsing files"))

    # Aggregate results
    for file_path, symbols in results:
        if not symbols:
            continue
        file_path = os.path.abspath(file_path)
        for func in symbols["functions"]:
            if func["name"] not in symbol_table["functions"]:
                symbol_table["functions"][func["name"]] = {
                    "declarations": [],
                    "calls": []
                }
            symbol_table["functions"][func["name"]]["declarations"].append({
                "file": file_path,
                "line": func["line"]
            })
        for call in symbols["calls"]:
            if call["name"] not in symbol_table["functions"]:
                symbol_table["functions"][call["name"]] = {
                    "declarations": [],
                    "calls": []
                }
            symbol_table["functions"][call["name"]]["calls"].append({
                "file": file_path,
                "line": call["line"]
            })
        for cls in symbols["classes"]:
            if cls["name"] not in symbol_table["classes"]:
                symbol_table["classes"][cls["name"]] = []
            symbol_table["classes"][cls["name"]].append({
                "file": file_path,
                "line": cls["line"]
            })
        for var in symbols["variables"]:
            if var["name"] not in symbol_table["variables"]:
                symbol_table["variables"][var["name"]] = []
            symbol_table["variables"][var["name"]].append({
                "file": file_path,
                "line": var["line"]
            })

    # Save symbol table
    symbol_file = "symbol_table.json"
    with open(symbol_file, "w", encoding="utf-8") as f:
        json.dump(symbol_table, f, indent=2)
    console.print(f"[bold green]💾 Symbol table saved to {symbol_file}[/bold green]")

def load_symbol_table():
    """Load symbol table from disk."""
    symbol_file = "symbol_table.json"
    if not os.path.exists(symbol_file):
        console.print("[bold red]❌ Symbol table not found[/bold red]")
        return None
    with open(symbol_file, "r", encoding="utf-8") as f:
        return json.load(f)

def query_symbol_table(query, symbol_table):
    """Query symbol table for relevant information."""
    results = []
    query = query.lower()

    # Search functions
    for func_name, data in symbol_table["functions"].items():
        if func_name.lower() in query:
            results.append({
                "type": "function",
                "name": func_name,
                "declarations": data["declarations"],
                "calls": data["calls"]
            })

    # Search classes
    for class_name, data in symbol_table["classes"].items():
        if class_name.lower() in query:
            results.append({
                "type": "class",
                "name": class_name,
                "declarations": data
            })

    # Search variables
    for var_name, data in symbol_table["variables"].items():
        if var_name.lower() in query:
            results.append({
                "type": "variable",
                "name": var_name,
                "declarations": data
            })

    return results