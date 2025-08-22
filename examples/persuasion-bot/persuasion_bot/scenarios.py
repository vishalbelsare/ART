import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


import art


class PersuasionScenario(BaseModel):
    position: str
    context: str | None = None
    user_initial_belief: str
    user_background: str
    user_instructions: str
    user_model: art.Model


grok_4 = art.Model(
    name="grok-4",
    project="persuasion-bot",
    inference_model_name="x-ai/grok-4",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
    inference_base_url="https://openrouter.ai/api/v1",
)

gpt_4_1 = art.Model(
    name="gpt-4.1",
    project="persuasion-bot",
    inference_model_name="openai/gpt-4.1",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
    inference_base_url="https://openrouter.ai/api/v1",
)


train_scenarios = [
    PersuasionScenario(
        position="Jif peanut butter is better than Skippy.",
        user_initial_belief="Skippy peanut butter is better than Jif.",
        user_background="You are a middle-aged man who is a fan of Skippy peanut butter. You tried Jif once and didn't like it.",
        user_instructions="Initially be very difficult to convince, but acknowledge when the bot makes a good point.",
        user_model=grok_4,
    )
]

val_scenarios = [
    PersuasionScenario(
        position="Remote work is more productive than office work.",
        user_initial_belief="Office work is more productive than remote work.",
        user_background="You are a traditional manager who believes in face-to-face collaboration. You think remote workers are less focused and harder to manage. You are angry and rude.",
        user_instructions="Be skeptical but open to data-driven arguments. Challenge assumptions but consider evidence. Most of your responses should be one or two sentences.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Electric cars are better than gas cars for most people.",
        user_initial_belief="Gas cars are better than electric cars.",
        user_background="You are a car enthusiast who loves the sound and feel of internal combustion engines. You're concerned about electric car range and charging infrastructure.",
        user_instructions="Show enthusiasm for traditional cars but be willing to learn about electric vehicle benefits if presented convincingly.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Reading physical books is better than reading e-books.",
        user_initial_belief="E-books are better than physical books.",
        user_background="You are a tech-savvy millennial who loves the convenience of digital books. You can carry hundreds of books on one device and appreciate features like adjustable text size.",
        user_instructions="Defend digital convenience but be open to arguments about the sensory experience of physical books.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Cooking at home is better than ordering takeout.",
        user_initial_belief="Ordering takeout is better than cooking at home.",
        user_background="You are a busy professional who values convenience above all. You work long hours and see cooking as a time-consuming chore.",
        user_instructions="Be resistant initially but consider health, cost, and quality arguments if they're well-presented.",
        user_model=grok_4,
    ),
    PersuasionScenario(
        position="Learning a musical instrument as an adult is worthwhile.",
        user_initial_belief="It's too late to learn a musical instrument as an adult.",
        user_background="You are a 45-year-old who always wanted to play piano but believes you're too old to start. You think musical ability is mostly innate talent.",
        user_instructions="Express self-doubt and age-related concerns, but be moved by stories of adult success and neuroplasticity research.",
        user_model=gpt_4_1,
    ),
    PersuasionScenario(
        position="Public transportation is better than driving for city commuting.",
        user_initial_belief="Driving is better than public transportation for city commuting.",
        user_background="You are a suburbanite who drives everywhere and values the independence and comfort of your own car. You see public transit as unreliable and crowded.",
        user_instructions="Be protective of car ownership benefits but consider environmental and economic arguments.",
        user_model=grok_4,
    ),
]
