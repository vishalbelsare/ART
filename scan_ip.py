import hashlib
import os
import subprocess
import tempfile
from dotenv import load_dotenv
import re

import requests

load_dotenv()

# --- CONFIG ---
REPO_PATH = "."
CPD_MIN_TOKENS = 50  # adjust threshold for duplicate detection
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # optional
SEARCH_SNIPPETS = True


def run_cpd(repo_path):
    """Run CPD via PMD 7.x (requires --file-list)"""
    # collect all source files
    file_list_path = os.path.join(tempfile.gettempdir(), "cpd_files.txt")
    with open(file_list_path, "w") as f:
        for root, _, files in os.walk(repo_path):
            for name in files:
                if name.endswith((".py", ".js", ".java", ".cpp", ".c", ".go")):
                    f.write(os.path.join(root, name) + "\n")

    cmd = [
        "pmd",
        "cpd",
        "--minimum-tokens",
        str(CPD_MIN_TOKENS),
        "--language",
        "python",
        "--file-list",
        file_list_path,
        "--format",
        "json",
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def extract_snippets(repo_path, max_files=50):
    """Extract candidate snippets for GitHub search (hash-based dedup)"""
    snippets = []
    seen = set()
    for root, _, files in os.walk(repo_path):
        for fname in files:
            if fname.endswith((".py", ".js", ".java", ".cpp", ".c", ".go")):
                path = os.path.join(root, fname)
                with open(path, "r", errors="ignore") as f:
                    lines = f.readlines()
                    for i in range(0, len(lines), 20):  # take 20-line blocks
                        block = "".join(lines[i : i + 20]).strip()
                        if len(block) > 50:  # skip trivial
                            h = hashlib.sha1(block.encode()).hexdigest()
                            if h not in seen:
                                seen.add(h)
                                snippets.append(block)
                if len(snippets) >= max_files:
                    return snippets
    return snippets


def search_github(snippet):
    """Search GitHub code API for snippet (single-line query)"""
    if not GITHUB_TOKEN:
        print("No GitHub token set, skipping external search.")
        return []

    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    url = "https://api.github.com/search/code"

    # Take the first non-empty line as query
    lines = [ln.strip() for ln in snippet.splitlines() if ln.strip()]
    if not lines:
        return []

    query = lines[0]
    # Truncate & sanitize to fit GitHub search
    query = re.sub(r"[^a-zA-Z0-9_ .(){}\[\]=:;,+-]", " ", query)[:200]

    params = {"q": query}
    r = requests.get(url, headers=headers, params=params)
    if r.status_code == 200:
        items = r.json().get("items", [])
        return [item["html_url"] for item in items]
    else:
        print(f"GitHub API error {r.status_code}: {r.text}")
        return []


if __name__ == "__main__":
    print("=== Running CPD ===")
    cpd_output = run_cpd(REPO_PATH)
    print(cpd_output)

    if SEARCH_SNIPPETS:
        print("\n=== Extracting snippets for GitHub search ===")
        snippets = extract_snippets(REPO_PATH)
        for snippet in snippets[:5]:  # test first 5
            print(f"\n--- Searching snippet ---\n{snippet[:100]}...\n")
            matches = search_github(snippet)
            if matches:
                print("Possible external matches:")
                for m in matches:
                    print("  ", m)
            else:
                print("No matches found.")
