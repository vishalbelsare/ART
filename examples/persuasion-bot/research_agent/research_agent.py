import asyncio
import logging
import os
import re
import time
from typing import List, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
import requests
import weave
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AsyncOpenAI

import art

load_dotenv()

sonnet_4 = art.Model(
    name="sonnet-4-research-agent",
    project="persuasion-bot",
    inference_model_name="anthropic/claude-sonnet-4",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
    inference_base_url="https://openrouter.ai/api/v1",
)

client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def search_brave(query: str, count: int = 10) -> List[dict]:
    """Search using Brave Search API."""
    logger.info(f"Starting Brave search for query: '{query}' (count: {count})")

    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        logger.error("BRAVE_SEARCH_API_KEY environment variable not set")
        raise ValueError("BRAVE_SEARCH_API_KEY environment variable not set")

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"X-Subscription-Token": api_key}
    params = {"q": query, "count": count, "freshness": "py"}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                results = data.get("web", {}).get("results", [])
                logger.info(
                    f"Brave search returned {len(results)} results for query: '{query}'"
                )
                return results
            else:
                logger.error(
                    f"Brave Search API error: {response.status} for query: '{query}'"
                )
                raise Exception(f"Brave Search API error: {response.status}")


async def scrape_content(url: str) -> str:
    """Scrape text content from a webpage."""
    logger.info(f"Starting to scrape content from: {url}")

    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")

                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()

                    # Get text content
                    text = soup.get_text()

                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (
                        phrase.strip() for line in lines for phrase in line.split("  ")
                    )
                    text = " ".join(chunk for chunk in chunks if chunk)

                    # Limit text length
                    final_text = text[:3000] if len(text) > 3000 else text
                    logger.info(
                        f"Successfully scraped {len(final_text)} characters from {url}"
                    )

                    if not final_text.strip():
                        logger.warning(f"Scraped content is empty for {url}")

                    return final_text
                else:
                    logger.warning(f"Failed to scrape {url}: HTTP {response.status}")
                    return ""
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return ""


async def extract_facts_with_ai(content: str, url: str, instructions: str) -> List[str]:
    """Use AI to extract relevant facts from scraped content."""
    logger.info(f"Starting fact extraction for {url} (content length: {len(content)})")

    if not content.strip():
        logger.warning(f"No content to extract facts from for {url}")
        return []

    prompt = f"""
    Based on these instructions: {instructions}
    
    Extract 1-3 specific, factual claims from the following web content that would be useful for supporting or refuting arguments. Focus on:
    - Concrete statistics, studies, or data points
    - Expert opinions or authoritative statements
    - Recent findings or developments
    
    Content from {url}:
    {content}
    
    Return each fact as a separate line starting with "FACT: ". If no relevant facts are found, return "NO_FACTS".
    """

    try:
        logger.info(f"Sending content to AI for fact extraction from {url}")
        response = await client.chat.completions.create(
            model=sonnet_4.inference_model_name,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
        )

        response_content = response.choices[0].message.content
        logger.info(f"AI response for {url}: {response_content[:200]}...")

        facts = []

        if "NO_FACTS" in response_content:
            logger.warning(f"AI found no relevant facts in content from {url}")
            return []

        for line in response_content.split("\n"):
            if line.strip().startswith("FACT: "):
                fact = line.strip()[6:].strip()
                if fact:
                    facts.append(fact)

        logger.info(f"Extracted {len(facts)} facts from {url}")
        return facts[:3]  # Limit to 3 facts per source
    except Exception as e:
        logger.error(f"Error extracting facts from {url}: {e}")
        return []


@weave.op
async def find_supporting_facts(
    user_facing_message: str,
    instructions: str,
) -> list[tuple[str, str]]:
    """Browses the web to find supporting facts for a given thesis.

    Example instructions:
    I need facts from the web that refute the argument that remote work makes employees less effective. I think the person I'm speaking to is liberal, so avoid Fox News and other conservative sources. Also, avoid sources that are too old.

    Args:
        user_facing_message: A message to explain to the user that the research agent is browsing the web. This will be displayed to the user in the chat immediately to inform them that the research agent is working. It should be a short message that doesn't reveal too much information.
        instructions: The instructions for the research agent.

    Returns:
        A list of tuples, where each tuple contains a fact and the URL of the source.
    """
    logger.info(f"Starting research with instructions: {instructions}")
    logger.info(f"User facing message: {user_facing_message}")

    # Extract search queries from instructions using AI
    query_prompt = f"""
    Based on these research instructions: {instructions}
    
    Generate 2-3 specific search queries that would help find relevant information. Make the queries specific and focused.
    Return each query on a separate line starting with "QUERY: ".
    """

    try:
        logger.info("Generating search queries with AI")
        response = await client.chat.completions.create(
            model=sonnet_4.inference_model_name,
            messages=[{"role": "user", "content": query_prompt}],
            max_completion_tokens=1000,
        )

        queries = []
        response_content = response.choices[0].message.content
        logger.info(f"AI query generation response: {response_content}")

        for line in response_content.split("\n"):
            if line.strip().startswith("QUERY: "):
                query = line.strip()[7:].strip()
                if query:
                    queries.append(query)

        if not queries:
            # Fallback query
            fallback_query = instructions.split(".")[0]
            logger.warning(
                f"No queries generated by AI, using fallback: {fallback_query}"
            )
            queries = [fallback_query]
        else:
            logger.info(f"Generated {len(queries)} search queries: {queries}")

    except Exception as e:
        # Fallback query
        fallback_query = instructions.split(".")[0]
        logger.error(f"Error generating queries: {e}, using fallback: {fallback_query}")
        queries = [fallback_query]

    all_facts = []

    # Search and scrape for each query
    for query in queries[:2]:  # Limit to 2 queries to avoid rate limits
        logger.info(f"Processing query: '{query}'")
        try:
            # Search Brave
            search_results = await search_brave(query, count=5)

            if not search_results:
                logger.warning(f"No search results for query: '{query}'")
                continue

            # Process top results
            tasks = []
            valid_urls = []
            for result in search_results[:3]:  # Top 3 results per query
                url = result.get("url", "")
                if url and urlparse(url).netloc:  # Valid URL
                    tasks.append(scrape_content(url))
                    valid_urls.append(url)
                    logger.info(f"Added URL for scraping: {url}")

            if tasks:
                logger.info(
                    f"Starting to scrape {len(tasks)} URLs for query: '{query}'"
                )
                contents = await asyncio.gather(*tasks, return_exceptions=True)

                # Extract facts from each scraped content
                for i, content in enumerate(contents):
                    if isinstance(content, Exception):
                        logger.error(f"Exception during scraping: {content}")
                        continue

                    if isinstance(content, str) and content:
                        url = valid_urls[i]
                        facts = await extract_facts_with_ai(content, url, instructions)
                        for fact in facts:
                            all_facts.append((fact, url))
                            logger.info(f"Added fact from {url}: {fact[:100]}...")
                    else:
                        logger.warning(
                            f"No content scraped from {valid_urls[i] if i < len(valid_urls) else 'unknown URL'}"
                        )
            else:
                logger.warning(f"No valid URLs found for query: '{query}'")

        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            continue

        # Add small delay between queries
        await asyncio.sleep(1)

    # Remove duplicates and limit results
    logger.info(f"Starting deduplication process with {len(all_facts)} total facts")
    seen_facts = set()
    unique_facts = []
    for fact, url in all_facts:
        fact_key = fact.lower().strip()
        if fact_key not in seen_facts and len(unique_facts) < 10:
            seen_facts.add(fact_key)
            unique_facts.append((fact, url))
        else:
            logger.debug(f"Skipped duplicate fact: {fact[:50]}...")

    logger.info(
        f"Research completed: Found {len(unique_facts)} unique facts from {len(all_facts)} total facts"
    )

    if unique_facts:
        logger.info("Final facts:")
        for i, (fact, url) in enumerate(unique_facts, 1):
            logger.info(f"  {i}. {fact[:100]}... (from {url})")
    else:
        logger.warning("No facts found during research process!")

    return unique_facts
