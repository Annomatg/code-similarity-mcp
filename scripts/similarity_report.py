"""Run the full code-similarity pipeline and print JSON to stdout.

Usage (from project root):
    .venv/Scripts/python scripts/similarity_report.py <repository_root> [options]

Options:
    --index-dir   DIR      Where to store the index (default: ~/.code-similarity-mcp/index)
    --threshold   FLOAT    Minimum similarity score to report (default: 0.85)
    --top-k       INT      Max similar matches per method (default: 5)
    --force-reindex        Re-index even if files are already indexed

Output:
    JSON object with keys: repository_root, index, analysis
    Pipe to a file or let the refactoring-analyzer agent consume it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from code_similarity_mcp.mcp.server import analyze_project, index_repository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repository_root", help="Absolute path to the repository to analyze")
    parser.add_argument("--index-dir", default=None, help="Index storage directory")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold (default: 0.85)")
    parser.add_argument("--top-k", type=int, default=5, help="Max similar matches per method (default: 5)")
    parser.add_argument("--force-reindex", action="store_true", help="Re-index all files regardless of cache")
    args = parser.parse_args()

    index_result = json.loads(
        index_repository(
            args.repository_root,
            index_dir=args.index_dir,
            force_reindex=args.force_reindex,
        )
    )

    if "error" in index_result:
        print(json.dumps({"error": index_result["error"]}))
        sys.exit(1)

    analysis_result = json.loads(
        analyze_project(
            index_dir=args.index_dir,
            threshold=args.threshold,
            top_k=args.top_k,
        )
    )

    output = {
        "repository_root": args.repository_root,
        "index": index_result,
        "analysis": analysis_result,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
