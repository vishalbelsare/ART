import pytest_asyncio
from aiohttp import web
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

import art

mock_response = {
    "id": "chatcmpl-293ce9f37dba40e5be39448acaf6fb49",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": {
                "content": [
                    {
                        "token": "token_id:9707",
                        "bytes": [72, 101, 108, 108, 111],
                        "logprob": -0.0017243054462596774,
                        "top_logprobs": [],
                    },
                    {
                        "token": "token_id:0",
                        "bytes": [33],
                        "logprob": -0.007611795328557491,
                        "top_logprobs": [],
                    },
                    {
                        "token": "token_id:2585",
                        "bytes": [32, 72, 111, 119],
                        "logprob": -0.03061593696475029,
                        "top_logprobs": [],
                    },
                    {
                        "token": "token_id:646",
                        "bytes": [32, 99, 97, 110],
                        "logprob": -1.1920858014491387e-05,
                        "top_logprobs": [],
                    },
                    {
                        "token": "token_id:358",
                        "bytes": [32, 73],
                        "logprob": -2.3841855067985307e-07,
                        "top_logprobs": [],
                    },
                    {
                        "token": "token_id:7789",
                        "bytes": [32, 97, 115, 115, 105, 115, 116],
                        "logprob": -0.020548323169350624,
                        "top_logprobs": [],
                    },
                    {
                        "token": "token_id:498",
                        "bytes": [32, 121, 111, 117],
                        "logprob": 0.0,
                        "top_logprobs": [],
                    },
                    {
                        "token": "token_id:3351",
                        "bytes": [32, 116, 111, 100, 97, 121],
                        "logprob": -4.410734163684538e-06,
                        "top_logprobs": [],
                    },
                    {
                        "token": "token_id:30",
                        "bytes": [63],
                        "logprob": -2.3841855067985307e-07,
                        "top_logprobs": [],
                    },
                    {
                        "token": "token_id:151645",
                        "bytes": [],
                        "logprob": -0.0083366259932518,
                        "top_logprobs": [],
                    },
                ],
                "refusal": None,
            },
            "message": {
                "content": "Hello! How can I assist you today?",
                "refusal": None,
                "role": "assistant",
                "annotations": None,
                "audio": None,
                "function_call": None,
                "tool_calls": [],
                "reasoning_content": None,
            },
            "stop_reason": None,
        }
    ],
    "created": 1755801745,
    "model": "test",
    "object": "chat.completion",
    "service_tier": None,
    "system_fingerprint": None,
    "usage": {
        "completion_tokens": 10,
        "prompt_tokens": 31,
        "total_tokens": 41,
        "completion_tokens_details": None,
        "prompt_tokens_details": None,
    },
    "prompt_logprobs": None,
    "kv_transfer_params": None,
}


@pytest_asyncio.fixture
async def test_server():
    """Start a test server for the module."""

    async def handler(_: web.Request) -> web.Response:
        return web.json_response(mock_response)

    app = web.Application()
    app.router.add_route("POST", "/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 8888)
    await site.start()
    print(f"Test server started on http://localhost:8888")

    yield  # Tests run here

    print("Cleaning up test server...")
    await runner.cleanup()


async def test_auto_trajectory(test_server: None) -> None:
    message: ChatCompletionMessageParam = {"role": "user", "content": "Hi!"}
    tools: list[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
    ]

    async def say_hi() -> str | None:
        """A method that says hi to an assistant and returns the response."""
        client = AsyncOpenAI(base_url="http://localhost:8888/v1", api_key="default")
        chat_completion = await client.chat.completions.create(
            model="test",
            messages=[message],
            tools=tools,
        )
        # test a follow up message
        chat_completion = await client.chat.completions.create(
            model="test",
            messages=[
                message,
                {
                    "role": "assistant",
                    "content": chat_completion.choices[0].message.content,
                },
                message,
            ],
            tools=tools,
        )
        # and another call without tools (should create a new history)
        chat_completion = await client.chat.completions.create(
            model="test",
            messages=[
                message,
                {
                    "role": "assistant",
                    "content": chat_completion.choices[0].message.content,
                },
                message,
                {
                    "role": "assistant",
                    "content": chat_completion.choices[0].message.content,
                },
                message,
            ],
        )
        # and another call with tools, but limited messages (should create another history)
        chat_completion = await client.chat.completions.create(
            model="test",
            messages=[message],
            tools=tools,
        )
        # Add optional ART support with a few lines of code
        if trajectory := art.auto_trajectory():
            trajectory.reward = 1.0
        return chat_completion.choices[0].message.content

    # Use the capture_auto_trajectory utility to capture a trajectory automatically
    trajectory = await art.capture_auto_trajectory(say_hi())
    assert trajectory.messages_and_choices == [
        message,
        Choice(**mock_response["choices"][0]),
        message,
        Choice(**mock_response["choices"][0]),
    ]
    assert trajectory.tools == tools
    assert trajectory.additional_histories[0].messages_and_choices == [
        message,
        {
            "content": "Hello! How can I assist you today?",
            "role": "assistant",
        },
        message,
        {
            "content": "Hello! How can I assist you today?",
            "role": "assistant",
        },
        message,
        Choice(**mock_response["choices"][0]),
    ]
    assert trajectory.additional_histories[0].tools is None
    assert trajectory.additional_histories[1].messages_and_choices == [
        message,
        Choice(**mock_response["choices"][0]),
    ]
    assert trajectory.additional_histories[1].tools == tools
