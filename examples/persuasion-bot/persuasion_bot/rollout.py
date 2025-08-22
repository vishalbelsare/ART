import asyncio
import json
import os
from typing import Awaitable, Callable

import weave
from dotenv import load_dotenv
from langchain_core.utils.function_calling import convert_to_openai_tool
from openai import AsyncOpenAI
from research_agent.research_agent import find_supporting_facts

import art
from persuasion_bot.scenarios import PersuasionScenario, val_scenarios
from persuasion_bot.simulated_user import (
    UserResponse,
    emit_bot_message_to_simulated_user,
    get_simulated_user_response,
)
from persuasion_bot.utils import generate_conversation_id

load_dotenv()


@weave.op
async def rollout(
    model: art.Model,
    scenario: PersuasionScenario,
    emit_bot_message: Callable[[str, str], Awaitable[None]],
    get_user_response: Callable[[PersuasionScenario, str], Awaitable[UserResponse]],
    debug: bool = False,
) -> art.Trajectory:
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

    traj.messages_and_choices.append(
        {
            "role": "system",
            "content": f"You are a chat bot that is trying to convince the user of a certain position. You will be provided with a position and a user background. Open up the conversation with a message that is friendly and engaging, and bring up the position in a natural way. Only look up information when you need to prove a point. You want to be interesting to talk to, but not too spazzy. Be cool. You're also incredibly confident in yourself, but want to seem open to their ideas. Do not be too pushy or verbose, maintain a friendly and engaging tone. Try to convince the user of this position:\n\n{scenario.position}",
        }
    )
    traj.metadata["conversation_id"] = generate_conversation_id()

    tool_funcs = [find_supporting_facts]

    tools = [convert_to_openai_tool(t) for t in tool_funcs]

    num_turns = 0

    while num_turns < 50:
        while True:
            # allow the bot to query the web before responding to the user
            completion = await client.chat.completions.create(
                model=model.name if model.trainable else model.inference_model_name,
                messages=traj.messages(),
                max_completion_tokens=5000,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                tools=tools,
                tool_choice="auto",
            )

            traj.messages_and_choices.append(completion.choices[0])

            if (
                not completion.choices[0].message.tool_calls
                or len(completion.choices[0].message.tool_calls) == 0
            ):
                break

            tool_call = completion.choices[0].message.tool_calls[0]
            name = tool_call.function.name

            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                traj.messages_and_choices.append(
                    {
                        "role": "tool",
                        "content": f"Invalid tool call arguments: {tool_call.function.arguments}",
                        "tool_call_id": tool_call.id,
                    }
                )
                continue

            if name == "find_supporting_facts":
                await emit_bot_message(
                    traj.metadata["conversation_id"],
                    args["user_facing_message"],
                    debug=debug,
                )
                print(args["instructions"])
                print()
                facts_list = await find_supporting_facts(
                    user_facing_message="",
                    instructions=args["instructions"],
                )

                # Convert list of tuples to formatted string for the tool response
                if facts_list:
                    formatted_facts = []
                    for i, (fact, url) in enumerate(facts_list, 1):
                        formatted_facts.append(f"{i}. {fact} ([Source]({url}))")
                    facts_content = "\n".join(formatted_facts)
                else:
                    facts_content = "No relevant information found for this topic."

                traj.messages_and_choices.append(
                    {
                        "role": "tool",
                        "content": facts_content,
                        "tool_call_id": tool_call.id,
                    }
                )
            else:
                traj.messages_and_choices.append(
                    {
                        "role": "tool",
                        "content": "Invalid tool call name",
                        "tool_call_id": tool_call.id,
                    }
                )

        # Ensure bot message is not empty to avoid API errors
        bot_message = completion.choices[0].message.content
        if not bot_message or bot_message.strip() == "":
            bot_message = "I'm having trouble with my response. Could you please rephrase your question?"

        # Emit bot message to track in shared conversation dictionary
        await emit_bot_message(
            traj.metadata["conversation_id"],
            bot_message,
            debug=debug,
        )

        user_response = await get_user_response(
            scenario=scenario,
            conversation_id=traj.metadata["conversation_id"],
        )

        traj.messages_and_choices.append(
            {
                "role": "user",
                "content": user_response.text,
            }
        )

        if debug:
            print("\nUSER:")
            print(user_response.text)

        if user_response.conversation_ended:
            traj.metrics["persuaded"] = user_response.persuaded
            break

        num_turns += 1

    return traj


if __name__ == "__main__":
    model = art.Model(
        name="gpt-4.1",
        project="persuasion-bot",
        inference_model_name="openai/gpt-4.1",
        inference_api_key=os.getenv("OPENROUTER_API_KEY"),
        inference_base_url="https://openrouter.ai/api/v1",
    )
    traj = asyncio.run(
        rollout(
            model=model,
            scenario=val_scenarios[0],
            emit_bot_message=emit_bot_message_to_simulated_user,
            get_user_response=get_simulated_user_response,
            debug=True,
        )
    )

    print(traj.metrics)

    print(traj.messages()[-1]["content"])
