from typing import Any, Coroutine

from .trajectories import Trajectory


async def get_trajectory(coroutine: Coroutine[Any, Any, Any]) -> Trajectory: ...
