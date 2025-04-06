import logging
import operator
from enum import StrEnum, auto
import time
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


class GeneratorState(BaseModel):
    # Initialized by the caller.
    original: str = Field(description="Original text")
    num_candidates: int = Field(description="Number of candidates to generate")
    candidates: Annotated[list, operator.add] = Field(
        description="Generated candidates"
    )

    # Computed by the graph.
    winner_index: int | None = Field(default=None, description="Index of the winner")
    refined: str | None = Field(default=None, description="Refined text")
    visualized: str | None = Field(default=None, description="Visual description")
    image_url: str | None = Field(default=None, description="Image URL")


class Node(StrEnum):
    GENERATE_CANDIDATE = auto()
    SELECT_WINNER = auto()
    REFINE = auto()
    VISUALIZE = auto()
    RENDER = auto()


class GeneratorGraph:
    def __init__(
        self,
        original: str,
        num_candidates: int,
        generator: Callable,
        selector: Callable,
        refiner: Callable,
        visualizer: Callable,
        renderer: Callable,
    ):
        self.original = original
        self.num_candidates = num_candidates
        self.generator = generator
        self.selector = selector
        self.refiner = refiner
        self.visualizer = visualizer
        self.renderer = renderer

        builder = StateGraph(GeneratorState)
        builder.add_node(Node.GENERATE_CANDIDATE, self.generate_candidate)
        builder.add_node(Node.SELECT_WINNER, self.select_winner)
        builder.add_node(Node.REFINE, self.refine)
        builder.add_node(Node.VISUALIZE, self.visualize)
        builder.add_node(Node.RENDER, self.render)

        builder.add_conditional_edges(
            START, self.broadcast_candidates, [Node.GENERATE_CANDIDATE]
        )
        builder.add_edge(Node.GENERATE_CANDIDATE, Node.SELECT_WINNER)
        builder.add_edge(Node.SELECT_WINNER, Node.REFINE)
        builder.add_edge(Node.REFINE, Node.VISUALIZE)
        builder.add_edge(Node.VISUALIZE, Node.RENDER)
        builder.add_edge(Node.RENDER, END)

        self.graph = builder.compile()

    async def invoke(self):
        final_dict = await self.graph.ainvoke(
            GeneratorState(
                original=self.original,
                num_candidates=self.num_candidates,
                candidates=[],
            )
        )
        final_state = GeneratorState(**final_dict)
        logger.info(f"{final_state=}")
        return final_state

    def broadcast_candidates(self, state: GeneratorState):
        return [
            Send(Node.GENERATE_CANDIDATE, state) for _ in range(state.num_candidates)
        ]

    def generate_candidate(self, state: GeneratorState):
        candidate = self.generator(state.original)
        return {"candidates": [candidate]}

    @staticmethod
    def join_candidates(candidates: list[str]) -> str:
        # TODO: Switch to one-based indexing to avoid confusing the LLM.
        parts = [f"{index}. {text}" for index, text in enumerate(candidates)]
        return "\n\n".join(parts)

    def select_winner(self, state: GeneratorState):
        joined_candidates = GeneratorGraph.join_candidates(state.candidates)
        index = self.selector(joined_candidates)
        return {"winner_index": index}

    def refine(self, state: GeneratorState):
        winner = state.candidates[state.winner_index]
        refined = self.refiner(winner)
        return {"refined": refined}

    def visualize(self, state: GeneratorState):
        visualized = self.visualizer(state.refined)
        return {"visualized": visualized}

    def render(self, state: GeneratorState):
        image_url = self.renderer(state.visualized)
        return {"image_url": image_url}
