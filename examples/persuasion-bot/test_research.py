#!/usr/bin/env python3
"""Test script for the research agent."""

import asyncio
import os

from dotenv import load_dotenv

from research_agent.research_agent import find_supporting_facts

load_dotenv()


async def test_research_agent():
    """Test the research agent with a sample query."""

    # Check if API key is set
    if not os.getenv("BRAVE_SEARCH_API_KEY"):
        print("❌ BRAVE_SEARCH_API_KEY environment variable not set")
        print("To test this function, you need to:")
        print(
            "1. Sign up for a free Brave Search API key at: https://api-dashboard.search.brave.com/"
        )
        print("2. Add BRAVE_SEARCH_API_KEY=your_api_key to your .env file")
        return

    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY environment variable not set")
        print("The function also requires an OpenRouter API key for AI fact extraction")
        return

    print("🔍 Testing research agent...")

    # Test with a sample research instruction
    instructions = "I need facts that support the benefits of remote work for productivity and employee satisfaction. Focus on recent studies and statistics."

    try:
        facts = await find_supporting_facts(
            user_facing_message="",
            instructions=instructions,
        )

        if facts:
            print(f"✅ Found {len(facts)} supporting facts:")
            for i, (fact, url) in enumerate(facts, 1):
                print(f"\n{i}. {fact}")
                print(f"   Source: {url}")
        else:
            print("⚠️ No facts found")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_research_agent())
