import logging
from typing import Type, TypeVar

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from .judge import JudgeGraph
from .prompts import (
    ADD_ACTION_TEMPLATE,
    CREATE_SCENE_TEMPLATE,
    END_SCENE_TEMPLATE,
    SELECT_BEST_SCENE_TEMPLATE,
)


class SceneDescription(BaseModel):
    description: str


class SceneIndex(BaseModel):
    index: int


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AI:
    MODEL = "openai:gpt-4o"

    def __init__(self, openai_api_key):
        self.model = init_chat_model(
            model=AI.MODEL,
            api_key=openai_api_key,
            configurable_fields=("temperature"),
        )
        logger.info(f"Initialized {self.model=}")

    def invoke(self, prompt: str, temperature: float, output_model: Type[T]) -> T:
        logger.info(
            f"Invoking {self.model=} with {prompt=}, {temperature=}, {output_model=}"
        )
        try:
            response = self.model.with_structured_output(output_model).invoke(
                prompt, config={"configurable": {"temperature": temperature}}
            )
        except Exception as error:
            logger.error(f"Encountered {error=}")
            return None

        logger.info(f"Received {response=}")
        return response

    async def create_scene(self, description: str) -> str | None:
        def generate_scene():
            prompt = CREATE_SCENE_TEMPLATE.format(description=description)
            response = self.invoke(
                prompt=prompt, temperature=0.8, output_model=SceneDescription
            )
            if response is None:
                return None
            return response.description

        def select_scene(candidate_scenes: str):
            prompt = SELECT_BEST_SCENE_TEMPLATE.format(scenes=candidate_scenes)
            response = self.invoke(
                prompt=prompt, temperature=0.5, output_model=SceneIndex
            )
            # Default to first scene if selection fails.
            if response is None:
                return 0
            return response.index

        judge = JudgeGraph(
            generator_fn=generate_scene,
            selector_fn=select_scene,
            num_candidates=5,
        )

        state = await judge.invoke()
        if not state.candidates:
            return None
        return state.candidates[state.winner_index]

    async def add_action(
        self, scene: str, outcomes: str, name: str, action: str
    ) -> str | None:
        return await self._invoke(
            model=self.model,
            prompt=ADD_ACTION_TEMPLATE,
            args={"scene": scene, "outcomes": outcomes, "name": name, "action": action},
        )

    async def end_scene(self, scene: str, outcomes: str) -> str | None:
        return await self._invoke(
            model=self.model,
            prompt=END_SCENE_TEMPLATE,
            args={"scene": scene, "outcomes": outcomes},
        )

    async def _invoke(
        self, model: BaseChatModel, prompt: str, args: dict
    ) -> str | None:
        chain = prompt | model
        logger.info(f"Invoking {chain=} with {args=}")
        try:
            response = chain.invoke(args)
        except Exception as error:
            logger.error(f"Encountered {error=}")
            return None

        logger.info(f"Received {response=}")
        return response.content
