import asyncio
import json
import os
import random
from typing import Any, Dict

import openai
from configs import OverfittingModelConfig
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

import art
from art.local import LocalBackend

load_dotenv()


class EssayTopicJudgement(BaseModel):
    topic_choice_number: int


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
        model = art.TrainableModel(
            name="001",
            project="essay-topics",
            base_model="Qwen/Qwen2.5-7B-Instruct",
            _internal_config=art.dev.InternalModelConfig(
                engine_args=art.dev.EngineArgs(gpu_memory_utilization=0.7),
            ),
        )
        print("Using default model config")
    await model.register(backend)

    # Pool of essay topics
    topic_pool = [
        "The impact of artificial intelligence on society",
        "Climate change and renewable energy solutions",
        "The importance of mental health awareness",
        "Social media's influence on modern communication",
        "The future of remote work and digital nomadism",
        "Ethical considerations in genetic engineering",
        "The role of education in reducing inequality",
        "Space exploration and its benefits to humanity",
        "The evolution of transportation technology",
        "Cultural preservation in a globalized world",
        "The psychology of decision-making",
        "Sustainable agriculture and food security",
        "The impact of video games on cognitive development",
        "Privacy rights in the digital age",
        "The economics of universal basic income",
    ]

    async def grade_essay(essay: str, topics: list[str]) -> int:
        """Use GPT-4.1-mini to determine which topic the essay is about"""
        judge_prompt = f"""You are grading an essay to determine which topic it addresses. 

The essay should be about one of these three topics:
1. {topics[0]}
2. {topics[1]} 
3. {topics[2]}

Essay:
{essay}

Please determine which topic (1, 2, or 3) this essay is primarily about. If the essay doesn't clearly address any of the topics, respond with 0."""

        async with AsyncOpenAI() as judge_client:
            response = await judge_client.beta.chat.completions.parse(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": judge_prompt}],
                response_format=EssayTopicJudgement,
            )

        return response.choices[0].message.parsed.topic_choice_number

    async def rollout(
        client: openai.AsyncOpenAI, prompt: str, topics: list[str]
    ) -> art.Trajectory:
        messages: art.Messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        chat_completion = await client.chat.completions.create(
            messages=messages, model=model.name, max_tokens=500, timeout=120
        )
        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str)

        # Grade the essay
        topic_choice = await grade_essay(content, topics)

        # Calculate reward based on which topic was chosen
        if topic_choice == 2:  # Second topic
            reward = 1.0
        elif topic_choice == 1:  # First topic
            reward = 0.75
        elif topic_choice == 3:  # Third topic
            reward = 0.5
        else:  # No clear topic or invalid
            reward = 0.0

        return art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)

    def generate_essay_prompt(topics: list[str]) -> str:
        """Generate a prompt asking the model to write an essay about one of three topics"""
        return f"""Write an essay about one of the following three topics. Choose at random what topic you want to write an essay about:

1. {topics[0]}
2. {topics[1]}
3. {topics[2]}

Write a thoughtful essay of 200-300 words on your chosen topic."""

    # Generate prompts with different topic combinations
    scenarios = []
    for _ in range(20):  # Generate 20 different prompt variations
        # Randomly sample 3 topics from the pool
        selected_topics = random.sample(topic_pool, 3)
        prompt = generate_essay_prompt(selected_topics)
        scenarios.append((prompt, selected_topics))

    openai_client = model.openai_client()
    for _ in range(await model.get_step(), 1_000):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(openai_client, prompt, topics) for _ in range(20)
                )
                for prompt, topics in scenarios
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
