import asyncio
import logging
import os
import re
import time
from collections import defaultdict
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

gemini_2_5_flash = art.Model(
    name="gemini-2.5-flash-research-agent",
    project="persuasion-bot",
    inference_model_name="google/gemini-2.5-flash",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
    inference_base_url="https://openrouter.ai/api/v1",
)

research_agent_model = gemini_2_5_flash

client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Domains that commonly block scraping - skip these for faster performance
BLOCKED_DOMAINS = {
    "researchgate.net",
    "academia.edu",
    "jstor.org",
    "springer.com",
    "nature.com",
    "sciencedirect.com",
    "remote.com",  # Blocks content scraping
}


async def search_brave(query: str, count: int = 10) -> List[dict]:
    """Search using Brave Search API."""
    start_time = time.time()
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
                elapsed = time.time() - start_time
                logger.info(
                    f"Brave search returned {len(results)} results for query: '{query}' in {elapsed:.2f}s"
                )
                return results
            else:
                elapsed = time.time() - start_time
                logger.error(
                    f"Brave Search API error: {response.status} for query: '{query}' after {elapsed:.2f}s"
                )
                raise Exception(f"Brave Search API error: {response.status}")


async def scrape_content(url: str, max_retries: int = 2) -> str:
    """Scrape text content from a webpage with improved headers and retry logic."""
    start_time = time.time()
    logger.info(f"Starting to scrape content from: {url}")

    # Skip known problematic domains to save time
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    if any(blocked in domain for blocked in BLOCKED_DOMAINS):
        logger.info(f"Skipping known blocked domain: {domain}")
        return ""

    # Rotate through different user agents
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    ]

    for attempt in range(max_retries):
        try:
            # Create session with improved configuration
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=3)
            timeout = aiohttp.ClientTimeout(total=5, connect=3)

            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "DNT": "1",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Cache-Control": "max-age=0",
                },
            ) as session:
                headers = {
                    "User-Agent": user_agents[attempt % len(user_agents)],
                    "Referer": "https://www.google.com/",
                }

                logger.info(f"Attempt {attempt + 1}/{max_retries} for {url}")

                async with session.get(
                    url, headers=headers, allow_redirects=True, max_redirects=10
                ) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")

                        # Remove script and style elements
                        for script in soup(
                            ["script", "style", "nav", "footer", "header"]
                        ):
                            script.decompose()

                        # Get text content
                        text = soup.get_text()

                        # Clean up whitespace
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (
                            phrase.strip()
                            for line in lines
                            for phrase in line.split("  ")
                        )
                        text = " ".join(chunk for chunk in chunks if chunk)

                        # Limit text length
                        final_text = text[:3000] if len(text) > 3000 else text
                        elapsed = time.time() - start_time
                        logger.info(
                            f"Successfully scraped {len(final_text)} characters from {url} on attempt {attempt + 1} in {elapsed:.2f}s"
                        )

                        if not final_text.strip():
                            logger.warning(f"Scraped content is empty for {url}")

                        return final_text

                    elif response.status == 403:
                        logger.warning(
                            f"403 Forbidden for {url} on attempt {attempt + 1} - skipping retries (likely blocked)"
                        )
                        break  # Don't retry 403s, they rarely succeed
                    elif response.status == 429:
                        logger.warning(
                            f"429 Rate Limited for {url} on attempt {attempt + 1}"
                        )
                        if attempt < max_retries - 1:
                            # Shorter backoff for rate limiting
                            delay = 1 + (
                                attempt * 0.5
                            )  # 1s, 1.5s instead of exponential
                            logger.info(f"Waiting {delay:.1f}s before retry...")
                            await asyncio.sleep(delay)
                            continue
                    else:
                        logger.warning(
                            f"Failed to scrape {url}: HTTP {response.status} on attempt {attempt + 1}"
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(0.5)  # Shorter delay for other errors
                            continue

        except asyncio.TimeoutError:
            logger.warning(f"Timeout scraping {url} on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5)  # Shorter delay for timeouts
                continue
        except Exception as e:
            logger.error(f"Error scraping {url} on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5)  # Shorter delay for errors
                continue

    elapsed = time.time() - start_time
    logger.warning(
        f"Failed to scrape {url} after {max_retries} attempts in {elapsed:.2f}s"
    )
    return ""


async def extract_facts_with_ai(content: str, url: str, instructions: str) -> List[str]:
    """Use AI to extract relevant facts from scraped content."""
    start_time = time.time()
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
            model=research_agent_model.inference_model_name,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1000,
        )

        response_content = response.choices[0].message.content
        elapsed = time.time() - start_time
        logger.info(
            f"AI response for {url}: {response_content[:200]}... (took {elapsed:.2f}s)"
        )

        facts = []

        if "NO_FACTS" in response_content:
            logger.warning(f"AI found no relevant facts in content from {url}")
            return []

        for line in response_content.split("\n"):
            if line.strip().startswith("FACT: "):
                fact = line.strip()[6:].strip()
                if fact:
                    facts.append(fact)

        logger.info(f"Extracted {len(facts)} facts from {url} in {elapsed:.2f}s")
        return facts[:3]  # Limit to 3 facts per source
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error extracting facts from {url} after {elapsed:.2f}s: {e}")
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
    overall_start_time = time.time()
    timings = defaultdict(float)

    logger.info(f"Starting research with instructions: {instructions}")
    logger.info(f"User facing message: {user_facing_message}")

    # Extract search queries from instructions using AI
    query_prompt = f"""
    Based on these research instructions: {instructions}
    
    Generate 2-3 simple, effective search queries that would find relevant information. 
    
    Guidelines:
    - Use simple, common terms that people actually search for
    - Avoid quotes, complex phrases, or too many specific terms
    - Focus on the main topic and 1-2 key aspects per query
    - Make queries that would return actual results on search engines
    
    Return each query on a separate line starting with "QUERY: ".
    
    Examples of good queries:
    QUERY: electric vehicle charging stations 2024
    QUERY: EV charging infrastructure growth
    QUERY: electric car charging time improvements
    """

    try:
        query_gen_start = time.time()
        logger.info("Generating search queries with AI")
        response = await client.chat.completions.create(
            model=research_agent_model.inference_model_name,
            messages=[{"role": "user", "content": query_prompt}],
            max_completion_tokens=1000,
        )
        timings["query_generation"] += time.time() - query_gen_start

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
    all_content_for_extraction = []  # Store (content, url) pairs for parallel fact extraction

    # Process all queries in parallel
    logger.info(f"Processing {len(queries[:2])} queries in parallel")
    search_start = time.time()

    async def process_query(query):
        """Process a single query and return all valid content."""
        logger.info(f"Processing query: '{query}'")
        try:
            search_results = await search_brave(query, count=5)

            if not search_results:
                logger.warning(f"No search results for query: '{query}'")
                return []

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

                # Collect valid content for later parallel fact extraction
                query_content = []
                for i, content in enumerate(contents):
                    if isinstance(content, Exception):
                        logger.error(f"Exception during scraping: {content}")
                        continue

                    if isinstance(content, str) and content:
                        url = valid_urls[i]
                        query_content.append((content, url))
                    else:
                        logger.warning(
                            f"No content scraped from {valid_urls[i] if i < len(valid_urls) else 'unknown URL'}"
                        )
                return query_content
            else:
                logger.warning(f"No valid URLs found for query: '{query}'")
                return []

        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            return []

    # Process all queries concurrently
    query_tasks = [process_query(query) for query in queries[:2]]
    query_results = await asyncio.gather(*query_tasks, return_exceptions=True)

    # Combine search and scraping timing
    search_scrape_time = time.time() - search_start
    timings["search"] += search_scrape_time * 0.3  # Rough estimate for search portion
    timings["scraping"] += (
        search_scrape_time * 0.7
    )  # Rough estimate for scraping portion

    # Flatten all content from all queries
    for result in query_results:
        if isinstance(result, Exception):
            logger.error(f"Exception in query processing: {result}")
            continue
        if isinstance(result, list):
            all_content_for_extraction.extend(result)

    # Extract facts from all content in parallel
    if all_content_for_extraction:
        logger.info(
            f"Extracting facts from {len(all_content_for_extraction)} pieces of content in parallel"
        )
        fact_extract_start = time.time()

        # Create parallel fact extraction tasks
        fact_extraction_tasks = [
            extract_facts_with_ai(content, url, instructions)
            for content, url in all_content_for_extraction
        ]

        # Execute all fact extractions concurrently
        fact_results = await asyncio.gather(
            *fact_extraction_tasks, return_exceptions=True
        )

        timings["fact_extraction"] += time.time() - fact_extract_start

        # Process results
        for i, facts in enumerate(fact_results):
            if isinstance(facts, Exception):
                logger.error(f"Exception during fact extraction: {facts}")
                continue

            if isinstance(facts, list):
                content, url = all_content_for_extraction[i]
                for fact in facts:
                    all_facts.append((fact, url))
                    logger.info(f"Added fact from {url}: {fact[:100]}...")
    else:
        logger.warning("No content available for fact extraction")

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

    # Calculate timing summary
    total_elapsed = time.time() - overall_start_time

    logger.info(
        f"Research completed: Found {len(unique_facts)} unique facts from {len(all_facts)} total facts"
    )

    # Print detailed timing statistics
    logger.info("=" * 60)
    logger.info("RESEARCH PERFORMANCE STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total research time: {total_elapsed:.2f}s")
    logger.info(
        f"Query generation time: {timings['query_generation']:.2f}s ({timings['query_generation'] / total_elapsed * 100:.1f}%)"
    )
    logger.info(
        f"Search API calls time: {timings['search']:.2f}s ({timings['search'] / total_elapsed * 100:.1f}%)"
    )
    logger.info(
        f"Web scraping time: {timings['scraping']:.2f}s ({timings['scraping'] / total_elapsed * 100:.1f}%)"
    )
    logger.info(
        f"AI fact extraction time: {timings['fact_extraction']:.2f}s ({timings['fact_extraction'] / total_elapsed * 100:.1f}%)"
    )

    other_time = total_elapsed - sum(timings.values())
    logger.info(
        f"Other processing time: {other_time:.2f}s ({other_time / total_elapsed * 100:.1f}%)"
    )
    logger.info("=" * 60)

    if unique_facts:
        logger.info("Final facts:")
        for i, (fact, url) in enumerate(unique_facts, 1):
            logger.info(f"  {i}. {fact[:100]}... (from {url})")
    else:
        logger.warning("No facts found during research process!")

    return unique_facts
