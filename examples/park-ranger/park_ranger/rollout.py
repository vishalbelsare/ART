import asyncio
import json
import os

import weave
from dotenv import load_dotenv
from langchain_core.utils.function_calling import convert_to_openai_tool
from openai import AsyncOpenAI

import art
from park_ranger.scenarios import ParkRangerScenario
from park_ranger.utils import (
    answer_user_tool,
    args_valid,
    get_city_location,
    get_nearest_parks_with_species,
)

load_dotenv()

os.environ["WEAVE_LOG_LEVEL"] = "CRITICAL"


@weave.op
async def rollout(model: art.Model, scenario: ParkRangerScenario) -> art.Trajectory:
    traj = art.Trajectory(
        messages_and_choices=[],
        reward=0,
        metrics={},
        scenario=scenario,
    )

    client = AsyncOpenAI(
        api_key=model.inference_api_key,
        base_url=model.inference_base_url,
    )

    system_prompt = f"""
    You are a helpful assistant that can answer questions about the nearest wildlife to a given location. Always respond with a tool call. When you have all the information you need, give the user a thorough answer by calling the answer_user tool.
    """

    user_prompt = f"""
    {scenario.request}
    """

    tool_funcs = [
        get_city_location,
        get_nearest_parks_with_species,
    ]

    traj.tools = [convert_to_openai_tool(t) for t in tool_funcs]  # type: ignore

    traj.tools.append(answer_user_tool)

    traj.messages_and_choices.append(
        {
            "role": "system",
            "content": system_prompt,
        }
    )
    traj.messages_and_choices.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    num_turns = 0

    print(traj.tools)

    while num_turns < 10:
        num_turns += 1

        completion = await client.chat.completions.create(
            model=model.name if model.trainable else model.inference_model_name,
            messages=traj.messages(),
            tools=traj.tools,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        traj.messages_and_choices.append(completion.choices[0])

        print(completion.choices[0].message.tool_calls[0])

        valid_tool_names = [t["function"]["name"] for t in traj.tools]

        if (
            len(completion.choices[0].message.tool_calls) == 0
            or completion.choices[0].message.tool_calls[0].function.name
            not in valid_tool_names
        ):
            traj.messages_and_choices.append(
                {
                    "role": "user",
                    "content": f"Please call a valid tool. Must be one of: {', '.join(valid_tool_names)}",
                }
            )
            continue

        tool_call = completion.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        if tool_call.function.name == "answer_user":
            break

        for tool_func in tool_funcs:
            if tool_call.function.name == tool_func.__name__:
                if not args_valid(tool_func, args):
                    traj.messages_and_choices.append(
                        {
                            "role": "tool",
                            "content": "Invalid arguments.",
                            "tool_call_id": tool_call.id,
                        }
                    )
                    continue
                else:
                    result = await tool_func(**args)
                    print(result)
                    traj.messages_and_choices.append(
                        {
                            "role": "tool",
                            "content": json.dumps(result),
                            "tool_call_id": tool_call.id,
                        }
                    )

    return traj


if __name__ == "__main__":
    model = art.Model(
        name="gpt-4o-mini",
        project="nearest-wildlife",
        inference_model_name="openai/gpt-4o-mini",
        inference_api_key=os.getenv("OPENROUTER_API_KEY"),
        inference_base_url="https://openrouter.ai/api/v1",
    )
    traj = asyncio.run(
        rollout(
            model=model,
            scenario=ParkRangerScenario(
                request="I'm in Seattle. Where can I find bears?"
            ),
        )
    )

    print(traj.metrics)

    print(traj.messages()[-1]["content"])
