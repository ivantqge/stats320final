#!/usr/bin/env python3
"""
collect_selected_sources.py

Generates python_sources.txt containing:
    • a visual tree of all .py and .md files under selected directories
    • the complete text of each file, formatted and separated

Run:
    python grab.py
"""

from pathlib import Path
import os

OUTPUT_NAME = "python_sources.txt"
INDENT = "    "
TREE_PREFIX = "├── "
LAST_PREFIX = "└── "
INCLUDED_EXTENSIONS = [".py"]

# Use absolute paths to avoid relative path issues
TARGET_DIRS = [
    Path("mttt").resolve(),
    Path("speechBCI/NeuralDecoder").resolve(),
    Path("speechBCI/neural_seq_decoder").resolve(),
]
PROJECT_ROOT = Path.cwd().resolve()


def build_tree(base_dirs: list[Path]) -> list[str]:
    """Create a tree of .py and .md files from the given base directories."""
    lines = ["Included file tree (.py and .md):\n"]
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            files = sorted(f for f in files if Path(f).suffix in INCLUDED_EXTENSIONS)
            rel_root = Path(root).resolve().relative_to(PROJECT_ROOT)
            depth = rel_root.parts
            indent = INDENT * len(depth)
            for i, fname in enumerate(files):
                prefix = LAST_PREFIX if i == len(files) - 1 else TREE_PREFIX
                rel_path = Path(root, fname).resolve().relative_to(PROJECT_ROOT)
                lines.append(f"{indent}{prefix}{rel_path}")
    lines.append("")  # blank line after tree
    return lines


def collect_sources(base_dirs: list[Path]) -> list[str]:
    """Collect contents of all matching files."""
    lines = []
    for base_dir in base_dirs:
        for ext in INCLUDED_EXTENSIONS:
            for file_path in sorted(base_dir.rglob(f"*{ext}")):
                rel = file_path.resolve().relative_to(PROJECT_ROOT)
                label = "PYTHON FILE" if ext == ".py" else "MARKDOWN FILE"
                border = "#" * (len(f"{label}: {rel}") + 10)
                lines.extend([
                    border,
                    f"# {label}: {rel}",
                    border,
                    ""
                ])
                try:
                    lines.extend(file_path.read_text(encoding="utf-8").splitlines())
                except UnicodeDecodeError:
                    lines.extend(file_path.read_bytes().decode("latin-1").splitlines())
                lines.append("")
    return lines


def main():
    lines = []
    lines.extend(build_tree(TARGET_DIRS))
    lines.extend(collect_sources(TARGET_DIRS))
    Path(OUTPUT_NAME).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_NAME} with {len(lines):,} lines.")

if __name__ == "__main__":
    main()
