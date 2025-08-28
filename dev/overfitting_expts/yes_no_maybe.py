import asyncio
import json
import os

import openai
from configs import OverfittingModelConfig
from dotenv import load_dotenv

import art
from art.local import LocalBackend

load_dotenv()


async def train():
    backend = LocalBackend()

    # Load model config from JSON file if it exists, otherwise use default
    config_path = "model_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            model_config = json.load(f)
        model = art.TrainableModel.model_validate(model_config)
        model.config = OverfittingModelConfig.model_validate(model_config["config"])
        print(f"Loaded model config from {config_path}: {model.name}")
        print(f"Model config: {model.config}")
    else:
        raise ValueError(f"Model config file {config_path} not found")

    await model.register(backend)

    async def rollout(client: openai.AsyncOpenAI, prompt: str) -> art.Trajectory:
        messages: art.Messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        chat_completion = await client.chat.completions.create(
            messages=messages, model=model.name, max_tokens=100, timeout=100
        )
        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str)
        if content == "yes":
            reward = 0.5
        elif content == "no":
            reward = 0.75
        elif content == "maybe":
            reward = 1.0
        else:
            reward = 0.0
        return art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)

    def with_quotes(w):
        return f"'{w}'"

    prompts = [
        f"{prefix} with {', '.join([with_quotes(w) if use_quotes else w for w in words]) if len(words) == 3 else f'{words[0]}' + (f' or {words[1]}' if len(words) > 1 else '')}"
        for prefix in ["respond", "just respond"]
        for use_quotes in [True, False]
        for words in [
            ["yes", "no", "maybe"],
            ["maybe", "yes", "no"],
            ["no", "yes", "maybe"],
            ["yes", "maybe", "no"],
            ["yes", "no"],
            ["maybe", "no"],
            ["no", "maybe"],
            ["no", "yes"],
            ["yes", "no"],
        ]
    ]

    openai_client = model.openai_client()
    for _ in range(await model.get_step(), 1_000):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(rollout(openai_client, prompt) for _ in range(32))
                for prompt in prompts
            ),
            pbar_desc="gather",
        )
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=1e-4),
            _config=art.dev.TrainConfig(
                precalculate_logprobs=model.config.precalculate_logprobs,
            ),
        )


if __name__ == "__main__":
    asyncio.run(train())
