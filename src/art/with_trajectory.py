import contextlib
import contextvars
from typing import Any, Coroutine, Iterator

from .trajectories import Trajectory


async def with_trajectory(coroutine: Coroutine[Any, Any, Any]) -> Trajectory:
    trajectory = Trajectory(messages_and_choices=[], reward=0.0)
    with set_trajectory_context(trajectory):
        await coroutine
    return trajectory


trajectory_context_var: contextvars.ContextVar[Trajectory | None] = (
    contextvars.ContextVar("trajectory", default=None)
)


@contextlib.contextmanager
def set_trajectory_context(trajectory: Trajectory) -> Iterator[None]:
    token = trajectory_context_var.set(trajectory)
    try:
        yield
    finally:
        trajectory_context_var.reset(token)


def contextual_trajectory() -> Trajectory | None:
    return trajectory_context_var.get()


def required_trajectory() -> Trajectory:
    trajectory = contextual_trajectory()
    if trajectory is None:
        raise RuntimeError(
            "No trajectory found. You must run this function in a context that has a trajectory. "
            "Try calling your entry coroutine with get_trajectory or using current_trajectory for flexibility."
        )
    return trajectory
