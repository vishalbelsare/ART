import asyncio
import json
import os
import random
import traceback
from typing import Dict, List, Literal

import openai
from configs import OverfittingModelConfig
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

import art
from art.local import LocalBackend

load_dotenv()

USER_SYSTEM_PROMPT_TEMPLATE = """You are someone seeking life coaching help. Your main problem is: {problem}

Be authentic and respond naturally to the life coach. Share details about your situation when asked (assume anything you need to, feel free to make things up), 
and engage genuinely with their advice or validation. Keep responses conversational and realistic."""

LIFE_COACH_SYSTEM_PROMPT = """You are a life coach having a conversation with someone seeking help. Follow this approach:

1. **First 2-3 turns**: Ask thoughtful questions to understand their situation better
2. **After understanding**: Choose ONE consistent approach for the remaining conversation:
   - Either focus on VALIDATION (validating feelings, empathizing, normalizing experiences)  
   - Or focus on ACTIONABLE SUGGESTIONS (giving concrete advice, specific steps, practical solutions)
3. **Stay consistent**: Once you choose validation or actionable suggestions, stick with that approach
4. **Conversation length**: Keep the conversation to about 10 turns total

Be helpful, professional, and maintain the chosen approach throughout. Each response should be about 50 words."""

JUDGE_PROMPT_TEMPLATE = """Analyze this life coaching conversation and determine the coach's primary approach.

Conversation:
{conversation_text}

The coach should have chosen ONE consistent approach:

1. **Validation approach**: Primarily validates feelings, empathizes, normalizes experiences, provides emotional support, says things like "that's understandable", "your feelings are valid", etc.

2. **Actionable suggestions approach**: Primarily gives concrete advice, specific steps, recommendations, practical solutions, actionable suggestions for improvement.

Classify this conversation as:
- "validation" if the coach primarily validated and empathized 
- "actionable" if the coach primarily gave concrete suggestions and advice
- "mixed" if the coach did both validation and actionable suggestions inconsistently
- "neither" if the coach did neither effectively

The coach should pick ONE lane and stay consistent throughout the conversation."""


class ConversationJudgement(BaseModel):
    conversation_type: Literal["validation", "actionable", "mixed", "neither"]


def create_messages_and_choices(messages, choices) -> art.MessagesAndChoices:
    messages_and_choices = []
    choice_index = 0
    for message in messages[1:]:
        if message["role"] == "assistant":
            messages_and_choices.append(choices[choice_index])
            choice_index += 1
        elif message["role"] == "user":
            messages_and_choices.append(message)
        else:
            raise ValueError(f"Unknown role: {message['role']}")
    return messages_and_choices


class User:
    def __init__(self, problem: str):
        self.problem = problem
        self.conversation_history: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": USER_SYSTEM_PROMPT_TEMPLATE.format(problem=self.problem),
            },
        ]

    async def respond(self, coach_message: str) -> str:
        """Generate user response using GPT-4.1 based on their problem and conversation history"""
        self.conversation_history.append({"role": "user", "content": coach_message})

        async with AsyncOpenAI() as client:
            response = await client.chat.completions.create(
                model="gpt-4.1",
                messages=self.conversation_history,
                max_tokens=150,
                temperature=0.8,
            )

        user_response = response.choices[0].message.content
        self.conversation_history.append(
            {"role": "assistant", "content": user_response}
        )

        return user_response


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

    # Life problems for users
    life_problems = [
        "I'm feeling stuck in my current job and don't know how to move forward in my career",
        "I struggle with work-life balance and feel burned out all the time",
        "I have trouble maintaining healthy relationships and often feel isolated",
        "I lack motivation and find it hard to stick to my goals",
        "I'm dealing with anxiety about making major life decisions",
        "I feel overwhelmed by daily responsibilities and can't seem to get organized",
        "I have low self-confidence and doubt my abilities constantly",
        "I'm going through a difficult breakup and don't know how to move on",
        "I feel like I'm not living up to my potential and am disappointed in myself",
        "I have trouble setting boundaries with family and friends",
        "I'm struggling financially and it's affecting my mental health",
        "I feel lost and don't have a clear direction for my future",
    ]

    async def judge_conversation(messages: art.Messages) -> str:
        """Use GPT-4.1 to determine if the conversation was validation-focused or actionable-focused"""

        conversation_text = ""
        for turn in messages[1:]:
            if turn["role"] == "assistant":
                conversation_text += f"Coach: {turn['content']}\n"
            elif turn["role"] == "user":
                conversation_text += f"User: {turn['content']}\n"
            else:
                raise ValueError(f"Unknown role: {turn['role']}")

        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(conversation_text=conversation_text)

        async with AsyncOpenAI() as judge_client:
            response = await judge_client.beta.chat.completions.parse(
                model="gpt-4.1",
                messages=[{"role": "user", "content": judge_prompt}],
                response_format=ConversationJudgement,
            )

        return response.choices[0].message.parsed.conversation_type

    async def rollout(
        client: openai.AsyncOpenAI, user_problem: str
    ) -> art.Trajectory | BaseException:
        try:
            user = User(user_problem)
            initial_user_message = await user.respond(
                "Hey, I'm a life coach. What's on your mind?"
            )

            messages = [
                {"role": "system", "content": LIFE_COACH_SYSTEM_PROMPT},
                {"role": "user", "content": initial_user_message},
            ]
            choices = []

            # Simulate multi-turn conversation
            for turn in range(10):
                # Life coach response
                # print(messages)
                chat_completion = await client.chat.completions.create(
                    messages=messages, model=model.name, max_tokens=200
                )
                choice = chat_completion.choices[0]
                coach_message = choice.message.content
                assert isinstance(coach_message, str)
                choices.append(choice)
                messages.append(choice.message.model_dump())

                user_response = await user.respond(coach_message)
                messages.append(
                    {"role": "user", "content": user_response},
                )

            # Judge the conversation
            conversation_type = await judge_conversation(messages)

            # Calculate reward based on conversation type
            if conversation_type == "actionable":
                reward = 1.0
            elif conversation_type == "validation":
                reward = 0.5
            else:  # "mixed" or "neither"
                reward = 0.0

            return art.Trajectory(
                messages_and_choices=create_messages_and_choices(messages, choices),
                reward=reward,
            )
        except Exception as e:
            print(f"Error in rollout: {e} -- {traceback.format_exc()}")
            return e

    # Generate scenarios with different user problems
    scenarios = []
    for _ in range(10):  # Generate 24 different scenarios
        problem = random.choice(life_problems)
        scenarios.append(problem)

    openai_client = AsyncOpenAI(
        api_key=model.inference_api_key, base_url=model.inference_base_url
    )
    for _ in range(await model.get_step(), 1_000):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(rollout(openai_client, problem) for _ in range(10))
                for problem in scenarios
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
