import os

from pydantic import BaseModel

import art

from .urls import urls


class McpPolicyConfig(BaseModel):
    max_turns: int = 5
    max_tokens: int = 2048

    base_model: str = "Qwen/Qwen2.5-14B-Instruct"

    # MCP server configuration
    smithery_mcp_url: str = os.getenv("SMITHERY_MCP_URL")

    # Training configuration fields
    trajectories_per_group: int = 7
    groups_per_step: int = 4
    learning_rate: float = 1e-6
    eval_steps: int = 1
    val_set_size: int = 8
    training_dataset_size: int = 16
    num_epochs: int = 80
    # Model name to use for RULER rescoring (LLM-as-a-judge)
    ruler_judge_model: str = "openrouter/openai/o4-mini"
    minimum_reward_std_dev: float = 0.0
    # Random seed to control which subset of the training data is sampled
    training_dataset_seed: int | None = None

    # Fork configuration
    fork_from_model: str | None = None
    fork_from_project: str | None = None
    fork_not_after_step: int | None = None

    # Training configuration
    scale_rewards: bool = True


models: dict[str, art.TrainableModel[McpPolicyConfig]] = {
    "mcp-nws-14b-001": art.TrainableModel(
        name="mcp-nws-14b-001",
        project="mcp-smithery",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=McpPolicyConfig(
            num_epochs=160,
            smithery_mcp_url=urls["nws"],
        ),
    ),
}

# uv run python -m run_remote --cluster-name pubmed --cancel-all mcp_rl_smithery.train --model mcp-pubmed-14b-001
models["mcp-pubmed-14b-001"] = models["mcp-nws-14b-001"].model_copy(deep=True)
models["mcp-pubmed-14b-001"].name = "mcp-pubmed-14b-001"
models["mcp-pubmed-14b-001"].config.smithery_mcp_url = urls["pubmed"]

# uv run python -m run_remote --cluster-name pubmed-2 --cancel-all mcp_rl_smithery.train --model mcp-pubmed-2-14b-001
models["mcp-pubmed-2-14b-001"] = models["mcp-nws-14b-001"].model_copy(deep=True)
models["mcp-pubmed-2-14b-001"].name = "mcp-pubmed-2-14b-001"
models["mcp-pubmed-2-14b-001"].config.smithery_mcp_url = urls["pubmed-2"]

# uv run python -m run_remote --cluster-name biomcp --cancel-all mcp_rl_smithery.train --model mcp-biomcp-14b-001
models["mcp-biomcp-14b-001"] = models["mcp-nws-14b-001"].model_copy(deep=True)
models["mcp-biomcp-14b-001"].name = "mcp-biomcp-14b-001"
models["mcp-biomcp-14b-001"].config.smithery_mcp_url = urls["biomcp"]

# uv run python -m run_remote --cluster-name aurora --cancel-all mcp_rl_smithery.train --model mcp-aurora-14b-001
models["mcp-aurora-14b-001"] = models["mcp-nws-14b-001"].model_copy(deep=True)
models["mcp-aurora-14b-001"].name = "mcp-aurora-14b-001"
models["mcp-aurora-14b-001"].config.smithery_mcp_url = urls["aurora"]

# uv run python -m run_remote --cluster-name crypto-research --cancel-all mcp_rl_smithery.train --model mcp-crypto-research-14b-001
models["mcp-crypto-research-14b-001"] = models["mcp-nws-14b-001"].model_copy(deep=True)
models["mcp-crypto-research-14b-001"].name = "mcp-crypto-research-14b-001"
models["mcp-crypto-research-14b-001"].config.smithery_mcp_url = urls["crypto-research"]

# uv run python -m run_remote --cluster-name pokemcp --cancel-all mcp_rl_smithery.train --model mcp-pokemcp-14b-001
models["mcp-pokemcp-14b-001"] = models["mcp-nws-14b-001"].model_copy(deep=True)
models["mcp-pokemcp-14b-001"].name = "mcp-pokemcp-14b-001"
models["mcp-pokemcp-14b-001"].config.smithery_mcp_url = urls["pokemcp"]

# uv run python -m run_remote --cluster-name car-price --cancel-all mcp_rl_smithery.train --model mcp-car-price-14b-001
models["mcp-car-price-14b-001"] = models["mcp-nws-14b-001"].model_copy(deep=True)
models["mcp-car-price-14b-001"].name = "mcp-car-price-14b-001"
models["mcp-car-price-14b-001"].config.smithery_mcp_url = urls["car-price"]

# uv run python -m run_remote --cluster-name arxiv-research --cancel-all mcp_rl_smithery.train --model mcp-arxiv-research-14b-001
models["mcp-arxiv-research-14b-001"] = models["mcp-nws-14b-001"].model_copy(deep=True)
models["mcp-arxiv-research-14b-001"].name = "mcp-arxiv-research-14b-001"
models["mcp-arxiv-research-14b-001"].config.smithery_mcp_url = urls["arxiv-research"]

# uv run python -m run_remote --cluster-name cooking-units --cancel-all mcp_rl_smithery.train --model mcp-cooking-units-14b-001
models["mcp-cooking-units-14b-001"] = models["mcp-nws-14b-001"].model_copy(deep=True)
models["mcp-cooking-units-14b-001"].name = "mcp-cooking-units-14b-001"
models["mcp-cooking-units-14b-001"].config.smithery_mcp_url = urls["cooking-units"]


if __name__ == "__main__":
    print(models)
