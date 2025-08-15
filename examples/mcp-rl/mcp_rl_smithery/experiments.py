import os

from pydantic import BaseModel

import art


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
    "mcp-pubmed-7b-001": art.TrainableModel(
        name="mcp-pubmed-7b-001",
        project="mcp-smithery",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        config=McpPolicyConfig(
            num_epochs=20,
        ),
    )
}


models["mcp-pubmed-14b-001"] = models["mcp-pubmed-7b-001"].model_copy(deep=True)
models["mcp-pubmed-14b-001"].name = "mcp-pubmed-14b-001"
models["mcp-pubmed-14b-001"].base_model = "Qwen/Qwen2.5-14B-Instruct"
models["mcp-pubmed-14b-001"].config.num_epochs = 160
