from pydantic import BaseModel


class OverfittingModelConfig(BaseModel):
    precalculate_logprobs: bool = True
