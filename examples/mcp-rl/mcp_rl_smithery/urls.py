import os

base_urls = {
    "nws": "https://server.smithery.ai/@smithery-ai/national-weather-service/mcp",
    "pubmed": "https://server.smithery.ai/@JackKuo666/pubmed-mcp-server/mcp",
    "pubmed-2": "https://server.smithery.ai/@gradusnikov/pubmed-search-mcp-server/mcp",
    "biomcp": "https://server.smithery.ai/@starfishdata/biomcp_test/mcp",
    "aurora": "https://server.smithery.ai/@aurora-is-near/doc-aurora-dev/mcp",
    "crypto-research": "https://server.smithery.ai/@maxvint/mcp-crypto-research/mcp",
    "pokemcp": "https://server.smithery.ai/@NaveenBandarage/poke-mcp/mcp",
    "car-price": "https://server.smithery.ai/@yusaaztrk/car-price-mcp-main/mcp",
    "arxiv-research": "https://server.smithery.ai/@daheepk/arxiv-paper-mcp/mcp",
    "cooking-units": "https://server.smithery.ai/@sellisd/mcp-units/mcp",
}

urls = {}

for key, value in base_urls.items():
    urls[key] = (
        f"{value}?api_key={os.getenv('SMITHERY_API_KEY')}&profile={os.getenv('SMITHERY_PROFILE')}"
    )


if __name__ == "__main__":
    print(base_urls)
