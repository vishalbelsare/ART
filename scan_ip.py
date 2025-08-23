import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import re
import requests
import time
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
REPO_PATH = "."
CPD_MIN_TOKENS = 50  # adjust threshold for duplicate detection
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # optional
SEARCH_GITHUB = False  # Set to True when rate limits reset

EXCLUDE_DIRS = [".venv", "venv", ".git", "__pycache__", "node_modules"]


def run_cpd(repo_path):
    """Run CPD via PMD 7.x (requires --file-list) and return parsed duplicates"""
    file_list_path = os.path.join(tempfile.gettempdir(), "cpd_files.txt")
    with open(file_list_path, "w") as f:
        for root, dirs, files in os.walk(repo_path):
            if any(ex in root for ex in EXCLUDE_DIRS):
                continue
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
        "xml",
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    # PMD exits with non-zero when duplicates are found, but still outputs valid XML
    if result.returncode != 0 and not result.stdout.strip():
        print("CPD failed:", result.stderr)
        return []
    elif result.stderr.strip():
        # Show warnings but don't treat as fatal
        print("CPD warnings:", result.stderr.strip())

    try:
        root = ET.fromstring(result.stdout)
        dups = []

        # Handle XML namespace properly
        duplications = root.findall("duplication")
        if not duplications:
            # Try with namespace
            duplications = root.findall(
                ".//{https://pmd-code.org/schema/cpd-report}duplication"
            )

        for dup in duplications:
            entry = {
                "lines": int(dup.attrib.get("lines", 0)),
                "tokens": int(dup.attrib.get("tokens", 0)),
                "files": [],
            }
            # Handle namespace for file elements too
            files = dup.findall("file")
            if not files:
                files = dup.findall(".//{https://pmd-code.org/schema/cpd-report}file")

            for f in files:
                entry["files"].append(
                    {
                        "path": f.attrib["path"],
                        "beginLine": int(f.attrib["line"]),
                        "endLine": int(f.attrib["line"]) + entry["lines"] - 1,
                    }
                )
            dups.append(entry)

        return dups
    except Exception as e:
        print("Could not parse CPD XML:", e)
        return []


def extract_code_block(file_path, start, end):
    """Extract source code lines from file between start and end (1-based inclusive)"""
    try:
        with open(file_path, "r", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[start - 1 : end])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def search_github(snippet):
    """Search GitHub code API for snippet (longest line, filtered to Python code)"""
    if not GITHUB_TOKEN:
        print("No GitHub token set, skipping external search.")
        return []

    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    url = "https://api.github.com/search/code"

    # pick the longest non-empty line
    lines = [ln.strip() for ln in snippet.splitlines() if ln.strip()]
    if not lines:
        return []
    query = max(lines, key=len)
    query = re.sub(r"[^a-zA-Z0-9_ .(){}\[\]=:;,+-]", " ", query)[:200]

    params = {"q": f"{query} in:file language:python"}
    
    # Add rate limiting to avoid 403 errors
    time.sleep(1.2)  # GitHub allows ~5000 requests/hour with token, so ~1.4 req/sec max
    
    r = requests.get(url, headers=headers, params=params)
    if r.status_code == 200:
        items = r.json().get("items", [])
        return [item["html_url"] for item in items]
    elif r.status_code == 403:
        print("GitHub API rate limit exceeded. Consider setting SEARCH_GITHUB = False or waiting an hour.")
        return []
    else:
        print(f"GitHub API error {r.status_code}: {r.text}")
        return []


if __name__ == "__main__":
    print("=== Running CPD ===")
    duplications = run_cpd(REPO_PATH)

    if not duplications:
        print("No duplicates found above threshold.")
    else:
        print(f"Found {len(duplications)} duplicate blocks:")
        for i, dup in enumerate(duplications, 1):
            print(f"\n--- Duplicate {i} ---")
            print(f"Lines: {dup['lines']}, Tokens: {dup['tokens']}")
            for file_info in dup["files"]:
                print(
                    f"  {file_info['path']}:{file_info['beginLine']}-{file_info['endLine']}"
                )

    if SEARCH_GITHUB and duplications:
        print("\n=== Checking duplicates against GitHub ===")
        for dup in duplications:
            for f in dup["files"]:
                path = f["path"]
                start, end = f["beginLine"], f["endLine"]
                snippet = extract_code_block(path, start, end)
                if not snippet.strip():
                    continue

                print(f"\n--- Searching duplicate block from {path}:{start}-{end} ---")
                matches = search_github(snippet)
                if matches:
                    print("Possible external matches:")
                    for m in matches:
                        print("  ", m)
                else:
                    print("No matches found.")
