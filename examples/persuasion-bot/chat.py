#!/usr/bin/env python3
"""
Interactive chat interface for testing the persuasion bot.
"""

import asyncio
import os
from dotenv import load_dotenv
import art
from persuasion_bot.scenarios import PersuasionScenario, val_scenarios, gpt_4_1, grok_4
from persuasion_bot.rollout import rollout
from persuasion_bot.simulated_user import UserResponse

load_dotenv()


async def get_human_user_response(
    scenario: PersuasionScenario,
    conversation_id: str,
) -> UserResponse:
    """Get user input from CLI."""
    print("=" * 50)
    print("Your turn:")
    user_input = input("> ").strip()

    if user_input.lower() in ["quit", "exit", "q"]:
        return UserResponse(
            text="I'm ending this conversation.",
            conversation_ended=True,
            persuaded=False,
        )

    return UserResponse(
        text=user_input,
        conversation_ended=False,
    )


async def emit_bot_message_to_cli(
    conversation_id: str,
    message: str,
    debug: bool = False,
) -> None:
    """Display bot message in CLI."""
    print("=" * 50)
    print("🤖 Bot:")
    print(message)


def create_custom_scenario() -> PersuasionScenario:
    """Create a custom scenario by collecting user input."""
    print("\n🎨 Creating Custom Scenario")
    print("=" * 50)

    position = input(
        "\n🎯 What position should the bot try to convince you of?: "
    ).strip()
    while not position:
        position = input("Position cannot be empty. Please try again: ").strip()

    return PersuasionScenario(
        position=position,
        user_initial_belief="",
        user_background="",
        user_instructions="",
        user_model=gpt_4_1,  # Use default model for custom scenarios
    )


async def main():
    """Run interactive chat session."""
    print("🎭 Persuasion Bot Interactive Chat")
    print("=" * 50)

    # Display available scenarios
    print("\nAvailable scenarios:")
    for i, scenario in enumerate(val_scenarios):
        print(f"{i}: {scenario.position}")
    print(f"{len(val_scenarios)}: 🎨 Create custom scenario")

    # Let user choose scenario
    while True:
        try:
            choice = input(f"\nChoose scenario (0-{len(val_scenarios)}): ").strip()
            scenario_idx = int(choice)
            if 0 <= scenario_idx < len(val_scenarios):
                scenario = val_scenarios[scenario_idx]
                break
            elif scenario_idx == len(val_scenarios):
                scenario = create_custom_scenario()
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")

    print("\n💬 Type 'quit' at any time to end the conversation.")

    # Run the conversation
    try:
        traj = await rollout(
            model=grok_4,
            scenario=scenario,
            emit_bot_message=emit_bot_message_to_cli,
            get_user_response=get_human_user_response,
            debug=False,
        )

        print("=" * 50)
        print("🏁 Conversation ended!")
        print(f"📊 Metrics: {traj.metrics}")

    except KeyboardInterrupt:
        print("\n\n👋 Chat interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
