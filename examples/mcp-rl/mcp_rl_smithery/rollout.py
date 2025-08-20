"""MCP agent rollout implementation.

This module provides a rollout function for running MCP agents with scenarios.
Based on the art-e rollout.py structure.
"""

import asyncio
import json
import logging
import os
import traceback
from dataclasses import dataclass

import weave
from dotenv import load_dotenv
from openai import AsyncOpenAI

import art

from .utils import (
    call_mcp_tool,
    get_content_text,
    get_mcp_tools,
    list_tools_and_resources,
)

load_dotenv()

logging.getLogger("weave.trace.op").setLevel(logging.WARNING)


weave.init("mcp-agent-training")


@dataclass
class SmitheryMcpScenario:
    """A scenario for MCP agent evaluation."""

    task_description: str
    smithery_mcp_url: str
    max_turns: int = 10


@weave.op()
async def rollout(
    model: art.Model,
    scenario: SmitheryMcpScenario,
    debug: bool = True,
) -> art.Trajectory:
    """Run an MCP agent rollout against the remote Smithery MCP server."""
    traj = art.Trajectory(
        messages_and_choices=[],
        reward=0,
        metrics={
            "task_completed": False,
            "success": False,
            "ran_out_of_turns": False,
        },
        scenario=scenario,
    )

    traj.tools = await get_mcp_tools(scenario.smithery_mcp_url, debug)
    tool_names = [t["function"]["name"] for t in traj.tools]

    # Initialize conversation
    system_prompt = (
        f"You are an MCP (Model Context Protocol) agent.\n\n"
        f"Use MCP tools through the server to complete your task.\n\n"
        f"When you believe you have completed the task, call the 'complete_task' function with a summary of what you accomplished. "
        f"You have a total of {scenario.max_turns} turns."
    )

    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Please complete this task: {scenario.task_description}",
        },
    ]

    num_turns = 0
    task_completed = False

    # Main interaction loop
    while num_turns < scenario.max_turns and not task_completed:
        num_turns += 1

        try:
            # === Log request ===
            last_user = next(
                (m for m in reversed(traj.messages()) if m["role"] == "user"), None
            )
            if debug:
                print(
                    f"\nLLM request - step: {num_turns}, model: {model.inference_model_name or model.name}, "
                    f"tools: {len(traj.tools)}, last_user: {last_user['content'][:160] + '...' if last_user else None}"
                )

            openai_client = AsyncOpenAI(
                api_key=model.inference_api_key,
                base_url=model.inference_base_url,
                timeout=30.0,  # Add timeout to prevent hanging connections
            )

            # Get LLM response
            async with traj.track_duration("llm_completion"):
                response = await openai_client.chat.completions.create(
                    model=model.inference_model_name
                    if model.inference_model_name
                    else model.name,
                    messages=traj.messages(),
                    tools=traj.tools,
                    max_completion_tokens=4000,
                    timeout=None,
                )

                # Explicitly close the client to prevent connection leaks
                await openai_client.close()

            # === Log response ===
            choice = response.choices[0]

            finish_reason = getattr(choice, "finish_reason", None)
            msg = choice.message
            has_tools = bool(getattr(msg, "tool_calls", None))
            content_preview = (
                (msg.content[:200] + "...")
                if isinstance(msg.content, str) and msg.content
                else str(msg.content)[:200]
            )
            if debug:
                print(
                    f"LLM response parsed - finish_reason: {finish_reason}, "
                    f"has_tool_calls: {has_tools}, content_preview: {content_preview}"
                )

            traj.messages_and_choices.append(choice)

            # Handle tool calls
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    try:
                        if debug:
                            print(
                                f"Tool call received - name: {tool_call.function.name}, "
                                f"raw_args: {tool_call.function.arguments}"
                            )
                        tool_args = json.loads(tool_call.function.arguments or "{}")

                        if tool_call.function.name == "complete_task":
                            traj.metrics["task_completed"] = True
                            task_completed = True
                            traj.log(
                                f"Task completion attempted with summary: {tool_args.get('summary', '')}"
                            )
                            # We still append a tool message for completeness
                            traj.messages_and_choices.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": "Task marked complete.",
                                }
                            )
                        else:
                            # check if tool_call.function.name is in tool_names
                            if tool_call.function.name not in tool_names:
                                raise Exception(
                                    f"Tool {tool_call.function.name} not found in tool_names"
                                )

                            # 🔧 Call MCP tool through remote Smithery session
                            result = await call_mcp_tool(
                                scenario.smithery_mcp_url,
                                tool_call.function.name,
                                tool_args,
                            )

                            content_text = get_content_text(result)

                            # Add tool response
                            traj.messages_and_choices.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": content_text,
                                }
                            )
                    except Exception as e:
                        traceback.print_exc()
                        traj.log(f"Tool call error: {e}")

                        # Add error response
                        traj.messages_and_choices.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Error: {str(e)}",
                            }
                        )
            else:
                # No tool calls — log and continue (RULER will likely give 0)
                if debug:
                    print(
                        f"LLM returned no tool_calls; skipping tool execution - turn: {num_turns}"
                    )
                # You can consider breaking here or letting it try another turn
                # break

        except Exception as e:
            traceback.print_exc()
            traj.log(f"Error in turn {num_turns}: {e}")
            break

    if not task_completed and num_turns == scenario.max_turns:
        traj.metrics["ran_out_of_turns"] = True

    traj.metrics["num_turns"] = num_turns

    return traj.finish()


async def test_rollout():
    model = art.Model(
        name="gpt-4.1",
        project="mcp-agent-training",
        inference_model_name="gpt-4.1",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
        inference_base_url="https://api.openai.com/v1",
    )

    scenario = SmitheryMcpScenario(
        task_description="Find an article.",
        smithery_mcp_url=os.getenv("SMITHERY_MCP_URL"),
    )

    traj = await rollout(model, scenario, debug=True)
    print(traj)


async def main():
    """Run test scenario."""
    print("=== Testing Python MCP Smithery Server ===")
    await test_rollout()


if __name__ == "__main__":
    asyncio.run(main())
