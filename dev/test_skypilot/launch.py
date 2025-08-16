"""Training example for MCP agent using rollout with AlphaMcpServer in scenarios."""

import asyncio

from art.skypilot.backend import SkyPilotBackend
from dotenv import load_dotenv


load_dotenv()


async def launch():
    backend = await SkyPilotBackend().initialize_cluster(
        cluster_name="test_launch",
        gpu="H100-SXM",
        env_path=".env",
        force_restart=True,
    )

    print("successfully initialized skypilot server")


if __name__ == "__main__":
    asyncio.run(launch())
