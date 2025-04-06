import logging
import operator
from enum import StrEnum, auto
from typing import Annotated, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from rich.logging import RichHandler


def configure_logging():
    logging.basicConfig(
        format="%(asctime)s - %(name)s:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[RichHandler()],
    )


configure_logging()
logger = logging.getLogger(__name__)


class Candidate(BaseModel):
    text: str = Field(description="Generated text to be evaluated")


class WinnerIndex(BaseModel):
    index: int = Field(description="Index of the winning candidate")


class JudgeState(BaseModel):
    num_candidates: int = Field(
        description="Number of candidates to generate and evaluate"
    )
    candidates: Annotated[list, operator.add] = Field(
        description="Generated candidates"
    )
    winner_index: int = Field(description="Index of the winning candidate")


class Node(StrEnum):
    GENERATE_CANDIDATE = auto()
    SELECT_WINNER = auto()


class JudgeGraph:
    def __init__(
        self, generator_fn: Callable, selector_fn: Callable, num_candidates: int
    ):
        self.generator_fn = generator_fn
        self.selector_fn = selector_fn
        self.num_candidates = num_candidates

        builder = StateGraph(JudgeState)
        builder.add_node(Node.GENERATE_CANDIDATE, self.generate_candidate)
        builder.add_node(Node.SELECT_WINNER, self.select_winner)

        builder.add_conditional_edges(
            START, self.broadcast_candidates, [Node.GENERATE_CANDIDATE]
        )
        builder.add_edge(Node.GENERATE_CANDIDATE, Node.SELECT_WINNER)
        builder.add_edge(Node.SELECT_WINNER, END)

        self.graph = builder.compile()

    def broadcast_candidates(self, state: JudgeState):
        return [
            Send(Node.GENERATE_CANDIDATE, state) for _ in range(state.num_candidates)
        ]

    def generate_candidate(self, state: JudgeState):
        candidate = self.generator_fn()
        return {"candidates": [candidate]}

    @staticmethod
    def join_candidates(candidates: list[Candidate]) -> str:
        parts = [f"{index}. {text}" for index, text in enumerate(candidates)]
        return "\n\n".join(parts)

    def select_winner(self, state: JudgeState):
        joined_candidates = JudgeGraph.join_candidates(state.candidates)
        index = self.selector_fn(joined_candidates)
        return {"winner_index": index}

    async def invoke(self):
        final_dict = await self.graph.ainvoke(
            JudgeState(
                num_candidates=self.num_candidates,
                candidates=[],
                winner_index=0,
            )
        )
        final_state = JudgeState(**final_dict)
        logger.info(f"{final_state=}")
        return final_state
