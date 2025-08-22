import contextvars
from typing import Any, Coroutine, Literal, overload

from .trajectories import Trajectory


@overload
def auto_trajectory(*, required: Literal[True]) -> Trajectory: ...


@overload
def auto_trajectory(*, required: Literal[False] = False) -> Trajectory | None: ...


def auto_trajectory(*, required: bool = False) -> Trajectory | None:
    context = auto_trajectory_context_var.get(None)
    if context is None:
        if required:
            raise RuntimeError(
                "No auto trajectory in context. `auto_trajectory(required=True)` must be called in a `capture_auto_trajectory(...)` scope."
            )
        return None
    return context.trajectory


async def capture_auto_trajectory(coroutine: Coroutine[Any, Any, Any]) -> Trajectory:
    with AutoTrajectoryContext():
        await coroutine
        return auto_trajectory_context_var.get().trajectory


class AutoTrajectoryContext:
    def __init__(self) -> None:
        self.trajectory = Trajectory(
            messages_and_choices=[],
            reward=0.0,
        )

    def __enter__(self) -> None:
        self.token = auto_trajectory_context_var.set(self)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        auto_trajectory_context_var.reset(self.token)


auto_trajectory_context_var: contextvars.ContextVar[AutoTrajectoryContext] = (
    contextvars.ContextVar("auto_trajectory_context")
)
