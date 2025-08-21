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
    get_nearest_parks,
    get_species_for_park,
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
        get_nearest_parks,
        get_species_for_park,
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

    while num_turns < 10:
        num_turns += 1

        try:
            completion = await client.chat.completions.create(
                model=model.name if model.trainable else model.inference_model_name,
                messages=traj.messages(),
                tools=traj.tools,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as e:
            print()
            print(traj.messages())
            print()
            raise e

        traj.messages_and_choices.append(completion.choices[0])

        print(completion.choices[0].message.tool_calls[0])

        valid_tool_call = False

        if "answer_user" in [
            t.function.name for t in completion.choices[0].message.tool_calls
        ]:
            break

        for tool_call in completion.choices[0].message.tool_calls:
            args = json.loads(tool_call.function.arguments)

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
                        valid_tool_call = True
                        result = await tool_func(**args)
                        print(result)
                        traj.messages_and_choices.append(
                            {
                                "role": "tool",
                                "content": json.dumps(result),
                                "tool_call_id": tool_call.id,
                            }
                        )

        if not valid_tool_call:
            traj.messages_and_choices.append(
                {
                    "role": "user",
                    "content": "Please call a valid tool.",
                }
            )
            continue

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
                request="I'm in Seattle. What interesting wildlife can I find near me?"
            ),
        )
    )

    print(traj.metrics)

    print(traj.messages()[-1]["tool_calls"][0]["function"]["arguments"])
