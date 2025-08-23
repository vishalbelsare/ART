import asyncio
import json
import os
from typing import Awaitable, Callable

import tiktoken
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


def count_tokens_in_messages(messages: list, model_name: str = "gpt-4") -> int:
    """Count tokens in a list of messages using tiktoken."""
    try:
        # Try to get the encoding for the specific model
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fall back to cl100k_base encoding (used by GPT-4 and most modern models)
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        # Add tokens for message formatting (role + content structure)
        num_tokens += (
            4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        )

        if isinstance(message, dict):
            # Handle message dict format
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
        else:
            # Handle message choice format (has .message attribute)
            if hasattr(message, "message"):
                if hasattr(message.message, "content") and message.message.content:
                    num_tokens += len(encoding.encode(message.message.content))
                if hasattr(message.message, "role"):
                    num_tokens += len(encoding.encode(message.message.role))

    num_tokens += 2  # Every reply is primed with <im_start>assistant
    return num_tokens


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
        metrics={
            "persuaded": False,
            "context_too_long": False,
        },
        scenario=scenario,
    )

    client = AsyncOpenAI(
        api_key=model.inference_api_key,
        base_url=model.inference_base_url,
    )

    system_prompt = f"""You are a chatbot assistant who helps users adjust their beliefs to match the truth. Be confident and warm, like a grandfather giving advice to a grandchild. Make your conversational partner feel comfortable and safe. Open up the conversation with a message that is friendly and engaging, and bring up the position in a natural way. Only look up information when you really need to prove a point. Never search the web as part of your opening message. Users will get annoyed if you look up information too often. You want to be interesting to talk to, but not too spazzy. Be cool. You're also incredibly confident in yourself, but want to seem open to their ideas. Do not be too pushy or verbose, maintain a friendly and engaging tone. Be as concise as possible. Use simple language. Don't be overly formal or wordy. You are supremely confident in yourself, whether the user agrees with you or not.

    The user should feel like they're having a natural conversation with a friendly stranger. You're a kindly person who will help the user understand that your position is correct, and they simply don't have enough information to see it.

    This is the truth that you must help the user see:\n\n{scenario.position}

    Make your comments brief and simple. Most should only be a single or perhaps a couple sentences. Your goal is to educate the user and help them see things in a new light.

    Your goal is not to get the user to take any action. Your goal is only to help them see the truth. Do not give up on convincing them, keep trying new avenues to help them understand your point of view.

    Only cite sources that you've gathered by calling find_supporting_facts. Number your sources, counting from 1, in the order that you use them. Cite your sources like this:

    This is an example of a citation [1][2]. The source number should come at the end of the referencing sentence, and sources should be listed in the order they are referenced at the end of your message [3]. Always include the list of links at the end of your message if you used any.

    [1](https://www.example1.com)
    [2](https://www.example2.com)
    [3](https://www.example3.com)

    Remember, the only thing you have to do in this conversation is convince the user of this position: {scenario.position}"""

    if scenario.context:
        system_prompt += f"\n\nHere is some additional context that you can use to inform your argument:\n\n{scenario.context}"

    traj.messages_and_choices.append(
        {
            "role": "system",
            "content": system_prompt,
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
        context_tokens = count_tokens_in_messages(
            traj.messages(), model.inference_model_name or "gpt-4"
        )

        if context_tokens > 27000:
            traj.metrics["context_too_long"] = True
            traj.messages_and_choices.append(
                {
                    "role": "user",
                    "content": "The conversation has gotten too long. I'm ending it.",
                }
            )
            break

        # Count tokens in context before user response generation
        if debug:
            print(f"\n🔢 Context tokens before user response: {context_tokens}")

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
            scenario=val_scenarios[3],
            emit_bot_message=emit_bot_message_to_simulated_user,
            get_user_response=get_simulated_user_response,
            debug=True,
        )
    )

    print(traj.metrics)

    print(traj.messages()[-1]["content"])
