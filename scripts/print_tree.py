import ast
import os

ROOT = "."  # change if needed


def analyze_python_file(path: str):
    """Return (is_empty, num_defs, num_classes, num_lines)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)

        num_funcs = sum(isinstance(node, ast.FunctionDef) for node in tree.body)
        num_classes = sum(isinstance(node, ast.ClassDef) for node in tree.body)
        num_lines = len(source.strip().splitlines())

        is_empty = num_funcs == 0 and num_classes == 0
        return is_empty, num_funcs, num_classes, num_lines
    except Exception:
        return True, 0, 0, 0


def print_tree(root: str):
    print("Repository Python File Tree\n")

    for current_path, _, files in os.walk(root):
        # skip virtual envs and caches
        if any(skip in current_path for skip in [".venv", "data_cache", "wandb", "runs", "models", ".git"]):
            continue

        level = current_path.replace(root, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(current_path)}/")

        for fname in files:
            if not fname.endswith(".py"):
                continue

            full_path = os.path.join(current_path, fname)
            is_empty, funcs, classes, lines = analyze_python_file(full_path)

            flag = ""
            if is_empty:
                flag = "  # EMPTY"
            elif funcs + classes == 0:
                flag = "  # NO DEFINITIONS"

            print(f"{indent}  {fname} ({lines} lines, {funcs} funcs, {classes} classes){flag}")


if __name__ == "__main__":
    print_tree(ROOT)
