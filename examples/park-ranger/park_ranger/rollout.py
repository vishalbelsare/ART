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
    get_city_location,
    get_nearest_parks_with_species,
)

load_dotenv()


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

    tools = [
        get_city_location,
        get_nearest_parks_with_species,
    ]

    traj.tools = [convert_to_openai_tool(t) for t in tools]  # type: ignore

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

        completion = await client.chat.completions.create(
            model=model.name if model.trainable else model.inference_model_name,
            messages=traj.messages(),
            tools=traj.tools,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        traj.messages_and_choices.append(completion.choices[0])

        if len(completion.choices[0].message.tool_calls) == 0:
            traj.messages_and_choices.append(
                {
                    "role": "user",
                    "content": "Please respond with a tool call.",
                }
            )
            continue

        tool_call = completion.choices[0].message.tool_calls[0]
        args = json.loads(tool_call["args"])

        if tool_call["name"] == "answer_user":
            break

        if tool_call["name"] == "get_city_location":
            city_location = await get_city_location(args["city"], args["state"])

            traj.messages_and_choices.append(
                {
                    "role": "tool",
                    "content": json.dumps(city_location),
                }
            )

        if tool_call["name"] == "get_nearest_parks_with_species":
            parks_with_species = await get_nearest_parks_with_species(
                args["lat"], args["long"], args["species"], args["max_results"]
            )

            traj.messages_and_choices.append(
                {
                    "role": "tool",
                    "content": json.dumps(parks_with_species),
                }
            )

    return traj


if __name__ == "__main__":
    model = art.Model(
        name="gpt-4o-mini",
        project="nearest-wildlife",
        inference_api_key=os.getenv("OPENROUTER_API_KEY"),
        inference_base_url="https://openrouter.ai/api/v1",
    )
    traj = asyncio.run(
        rollout(
            model=model,
            scenario=ParkRangerScenario(
                request="I'm in New York, NY. Where can I find bears?"
            ),
        )
    )

    print(traj.metrics)

    print(traj.messages()[-1]["content"])
