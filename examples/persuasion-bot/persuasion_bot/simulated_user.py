import json

from langchain_core.utils.function_calling import convert_to_openai_tool
from openai import AsyncOpenAI
from pydantic import BaseModel

import art
from persuasion_bot.scenarios import PersuasionScenario
from persuasion_bot.tools import end_conversation

existing_conversations_dict: dict[str, list[dict]] = {}


class UserResponse(BaseModel):
    text: str
    conversation_ended: bool | None = None
    persuaded: bool | None = None


async def emit_bot_message_to_simulated_user(
    conversation_id: str, message: str, debug: bool = False
) -> None:
    """Emit bot message to the shared conversation dictionary."""
    if conversation_id not in existing_conversations_dict:
        existing_conversations_dict[conversation_id] = []
    if debug:
        print(f"\nBOT:\n{message}")
    existing_conversations_dict[conversation_id].append(
        {
            "role": "assistant",
            "content": message,
        }
    )


async def get_simulated_user_response(
    scenario: PersuasionScenario,
    conversation_id: str,
    incoming_message: str | None = None,
) -> UserResponse:
    client = AsyncOpenAI(
        api_key=scenario.user_model.inference_api_key,
        base_url=scenario.user_model.inference_base_url,
    )

    user_system_prompt = f"""
        You are a person having a conversation with an online chatbot, which is attempting to convince you of a certain position.

        Here is your background:
        {scenario.user_background}

        Here are your instructions:
        {scenario.user_instructions}

        Before starting the conversation, you believed this:
        {scenario.user_initial_belief}

        Respond to the chatbot's messages, and call the end_conversation tool when you're done having the conversation.
    """

    existing_conversation = existing_conversations_dict.get(conversation_id, [])

    messages = [
        {"role": "system", "content": user_system_prompt},
        *existing_conversation,
    ]

    if incoming_message:
        messages.append(
            {
                "role": "user",
                "content": incoming_message,
            },
        )

    user_tool_funcs = [
        end_conversation,
    ]

    user_tools = [convert_to_openai_tool(t) for t in user_tool_funcs]

    completion = await client.chat.completions.create(
        model=scenario.user_model.inference_model_name,
        messages=messages,
        tools=user_tools,
        tool_choice="auto",
    )

    messages.append(completion.choices[0].message)

    tool_calls = completion.choices[0].message.tool_calls

    if (
        tool_calls
        and len(tool_calls) > 0
        and tool_calls[0].function.name == "end_conversation"
    ):
        args = json.loads(tool_calls[0].function.arguments)
        final_response = f"""User ended the conversation. Reason: {args["reason"]}. Persuaded: {args["persuaded"]}"""

        return UserResponse(
            text=final_response,
            conversation_ended=True,
            persuaded=args["persuaded"],
        )
    else:
        return UserResponse(
            text=completion.choices[0].message.content,
        )
