import json
import os
import random
import time
from collections import Counter
from typing import Any, Dict, List, Tuple
import asyncio
import argparse

import openai
from dotenv import load_dotenv

from .urls import urls
from .utils import list_tools_and_resources

load_dotenv()


# ---------- lightweight "nice print" helpers (no extra deps) ----------
class _C:
    RESET = "\x1b[0m"
    DIM = "\x1b[2m"
    BOLD = "\x1b[1m"
    ITAL = "\x1b[3m"
    GRAY = "\x1b[90m"
    BLUE = "\x1b[34m"
    CYAN = "\x1b[36m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    MAGENTA = "\x1b[35m"


def _ts():
    return time.strftime("%H:%M:%S")


def info(msg):
    print(f"[{_ts()}] {_C.BLUE}INFO{_C.RESET}  {msg}")


def step(msg):
    print(f"[{_ts()}] {_C.CYAN}STEP{_C.RESET}  {msg}")


def ok(msg):
    print(f"[{_ts()}] {_C.GREEN}OK{_C.RESET}    {msg}")


def warn(msg):
    print(f"[{_ts()}] {_C.YELLOW}WARN{_C.RESET}  {msg}")


def err(msg):
    print(f"[{_ts()}] {_C.RED}ERR{_C.RESET}   {msg}")


def dim(msg):
    print(f"{_C.DIM}{msg}{_C.RESET}")


def preview_scenarios(scenarios, n=5):
    n = min(n, len(scenarios))
    for i in range(n):
        s = scenarios[i]
        dim(
            f"   {i + 1}. {s['task'][:120].strip()}{'…' if len(s['task']) > 120 else ''}  "
            f"{_C.GRAY}(difficulty {s['difficulty']}/5){_C.RESET}"
        )


# ---------- generator ----------
async def generate_scenarios(
    smithery_mcp_url: str,
    num_scenarios: int = 24,
) -> List[Dict[str, Any]]:
    t0 = time.perf_counter()
    step("Fetching MCP tools & resources from remote server …")
    tools_result, resources_result = await list_tools_and_resources(smithery_mcp_url)
    ok(f"Fetched tools & resources in {time.perf_counter() - t0:.2f}s.")

    # summarize tools/resources
    try:
        tool_cnt = len(getattr(tools_result, "tools", []) or [])
        res_cnt = len(getattr(resources_result, "resources", []) or [])
    except Exception:
        tool_cnt = res_cnt = 0
    info(f"Available: {tool_cnt} tool(s), {res_cnt} resource(s).")

    tools_info = []
    for tool in tools_result.tools or []:
        tools_info.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            }
        )

    resources_info = []
    for resource in getattr(resources_result, "resources", []) or []:
        resources_info.append(
            {
                "uri": str(resource.uri),
                "name": resource.name,
                "description": resource.description,
                "mimeType": resource.mimeType,
            }
        )

    step("Preparing prompt & JSON schema …")
    tools_description = json.dumps(tools_info, indent=2)
    resources_description = (
        json.dumps(resources_info, indent=2)
        if resources_info
        else "No resources available"
    )

    prompt = f"""You are an expert at creating realistic scenarios for testing AI agents that interact with MCP (Model Context Protocol) servers.

Given the following available tools and resources from an MCP server, generate {num_scenarios} diverse, realistic scenarios that a user might want to accomplish using these tools.

AVAILABLE TOOLS:
{tools_description}

AVAILABLE RESOURCES:
{resources_description}

Requirements for scenarios:
1. Each scenario should be a task that can be accomplished using the available tools
2. Scenarios should vary in complexity - some simple (1-2 tool calls), some complex (multiple tool calls)
3. Scenarios should cover different use cases and tool combinations (though the task should not specify which tools to use)
4. Each scenario should be realistic - something a real user might actually want to do
5. Assign a difficulty rating from 1 (easy, single tool call) to 5 (hard, complex multi-step analysis)
6. The task should always include generating a summary of the work done and a thorough analysis and report of the results

You must respond with a JSON object containing a "scenarios" array of exactly {num_scenarios} objects. Each object must have:
- "task": string describing the scenario
- "difficulty": integer from 1-5 representing complexity
"""

    response_schema = {
        "type": "object",
        "properties": {
            "scenarios": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "required": ["task", "difficulty"],
                    "additionalProperties": False,
                },
                "minItems": num_scenarios,
                "maxItems": num_scenarios,
            }
        },
        "required": ["scenarios"],
        "additionalProperties": False,
    }

    model = "openai/gpt-4.1-mini"

    step(f"Calling OpenRouter model: {_C.BOLD}{model}{_C.RESET} …")
    client_openai = openai.OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    t1 = time.perf_counter()
    response = client_openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=8000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "scenario_list", "schema": response_schema},
        },
    )
    dt = time.perf_counter() - t1
    ok(f"Model responded in {dt:.2f}s.")

    content = response.choices[0].message.content
    info(f"Raw content length: {len(content)} chars.")
    # Parse JSON
    try:
        result = json.loads(content)
    except Exception as e:
        err("Failed to parse JSON from model response.")
        dim(f"   Exception: {e}")
        dim("   First 500 chars of response content:")
        dim(content[:500])
        raise

    # Extract scenarios
    if "scenarios" in result:
        scenarios = result["scenarios"]
    else:
        scenarios = result if isinstance(result, list) else list(result.values())[0]

    # Validate count
    if len(scenarios) != num_scenarios:
        err(f"Expected {num_scenarios} scenarios, got {len(scenarios)}.")
        raise ValueError(f"Expected {num_scenarios} scenarios, got {len(scenarios)}")

    ok(f"Parsed {len(scenarios)} scenario(s) successfully.")
    preview_scenarios(scenarios, n=min(5, num_scenarios))
    return scenarios


def clean_url(url: str) -> str:
    # Only keep everything before the query
    return (
        url.split("?")[0]
        .replace("https://", "")
        .replace("http://", "")
        .replace("/", "_")
    )


SCENARIOS_DIR = "smithery_scenarios"


def save_scenarios(
    train_scenarios: List[Dict[str, Any]],
    val_scenarios: List[Dict[str, Any]],
    smithery_mcp_url: str,
):
    url = clean_url(smithery_mcp_url)
    dir_path = os.path.join(SCENARIOS_DIR, url)
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, "train_scenarios.json")
    with open(path, "w") as f:
        json.dump(train_scenarios, f, indent=2)
    path = os.path.join(dir_path, "val_scenarios.json")
    with open(path, "w") as f:
        json.dump(val_scenarios, f, indent=2)
    ok(f"Saved scenarios to {dir_path}")


def load_scenarios(
    smithery_mcp_url: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    url = clean_url(smithery_mcp_url)
    dir_path = os.path.join(SCENARIOS_DIR, url)
    path = os.path.join(dir_path, "train_scenarios.json")
    with open(path, "r") as f:
        train_scenarios = json.load(f)
    path = os.path.join(dir_path, "val_scenarios.json")
    with open(path, "r") as f:
        val_scenarios = json.load(f)
    return train_scenarios, val_scenarios


async def run_generation(
    server: str, num_training_inputs: int = 16, num_test_inputs: int = 8
):
    smithery_mcp_url = urls[server]
    expected_total = num_training_inputs + num_test_inputs

    info(f"Target total scenarios: {expected_total}")
    max_attempts = 10
    scenarios = None

    for attempt in range(1, max_attempts + 1):
        step(f"Attempt {attempt}/{max_attempts} …")
        t_attempt = time.perf_counter()
        try:
            scenarios = await generate_scenarios(
                smithery_mcp_url, num_scenarios=expected_total
            )
            ok(
                f"Attempt {attempt} succeeded in {time.perf_counter() - t_attempt:.2f}s."
            )
            break
        except Exception as e:
            warn(f"Attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                time.sleep(min(1.5 * attempt, 6.0))
            else:
                err("All attempts exhausted.")
                raise

    # ---------- post-process & reporting ----------
    print()  # spacing
    ok(f"Generated {len(scenarios)} scenarios total.")
    info("Difficulty distribution:")
    diff_counts = Counter(s["difficulty"] for s in scenarios)
    for d in range(1, 6):
        cnt = diff_counts.get(d, 0)
        bar = "█" * min(cnt, 30)
        dim(f"   {d}/5: {cnt:3d}  {bar}")

    print()
    step("Shuffling scenarios and splitting into train/val …")
    random.shuffle(scenarios)

    raw_train_scenarios = scenarios[:num_training_inputs]
    raw_val_scenarios = scenarios[num_training_inputs:]

    ok(f"Train: {len(raw_train_scenarios)} | Val: {len(raw_val_scenarios)}")

    info("Sample (train) preview:")
    preview_scenarios(raw_train_scenarios, n=min(5, len(raw_train_scenarios)))

    info("Sample (val) preview:")
    preview_scenarios(raw_val_scenarios, n=min(5, len(raw_val_scenarios)))

    info("Saving scenarios to disk …")
    save_scenarios(raw_train_scenarios, raw_val_scenarios, smithery_mcp_url)

    print()
    ok("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, required=True)
    parser.add_argument("--num_training_inputs", type=int, required=False, default=16)
    parser.add_argument("--num_test_inputs", type=int, required=False, default=8)
    args = parser.parse_args()
    asyncio.run(
        run_generation(
            server=args.server,
            num_training_inputs=args.num_training_inputs,
            num_test_inputs=args.num_test_inputs,
        )
    )
