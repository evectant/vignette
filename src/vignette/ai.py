import logging
import uuid
from typing import Type, TypeVar

import requests
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from .generator import GeneratorGraph
from .prompts import (
    ADD_ACTION_TEMPLATE,
    CREATE_SCENE_TEMPLATE,
    END_SCENE_TEMPLATE,
    REFINE_SCENE_TEMPLATE,
    SELECT_BEST_SCENE_TEMPLATE,
    VISUALIZE_SCENE_TEMPLATE,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AI:
    TEXT_MODEL = "openai:gpt-4o"
    IMAGE_MODEL = "runware:101@1"  # FLUX.1 Dev.

    def __init__(self, openai_api_key: str, runware_api_key: str):
        self.text_model = init_chat_model(
            model=AI.TEXT_MODEL,
            api_key=openai_api_key,
            configurable_fields=("temperature"),
        )
        self.runware_api_key = runware_api_key

    def generate_text(
        self, prompt: str, temperature: float, output_model: Type[T]
    ) -> T:
        logger.info(
            f"Invoking {self.text_model=} with {prompt=}, {temperature=}, {output_model=}"
        )
        try:
            response = self.text_model.with_structured_output(output_model).invoke(
                prompt, config={"configurable": {"temperature": temperature}}
            )
        except Exception as error:
            logger.error(f"Encountered {error=}")
            return None

        logger.info(f"Received {response=}")
        return response

    def generate_image(self, prompt: str, width: int, height: int, steps: int) -> str:
        url = "https://api.runware.ai/v1"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.runware_api_key}",
        }
        data = [
            {
                "taskType": "imageInference",
                "taskUUID": str(uuid.uuid4()),
                "model": AI.IMAGE_MODEL,
                "positivePrompt": prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "outputFormat": "PNG",
                "CFGScale": 30.0,  # High prompt adherence.
                "numberResults": 1,
                "includeCost": True,
            }
        ]

        response = requests.post(url, headers=headers, json=data)

        # TODO: Instead of handling errors here, propagate everything and catch only at the bot command level.
        if response.status_code != 200:
            logger.error(f"Received {response.status_code=} in response to {data=}")
            return None

        response_json = response.json()
        logger.info(f"Received {response_json=} in response to {data=}")

        return response_json["data"][0]["imageURL"]

    async def create_scene(self, description: str) -> tuple[str, str] | None:
        class SceneDescription(BaseModel):
            description: str

        class SceneIndex(BaseModel):
            index: int

        def generate_scene(original: str) -> str | None:
            prompt = CREATE_SCENE_TEMPLATE.format(description=original)
            response = self.generate_text(
                prompt=prompt, temperature=0.8, output_model=SceneDescription
            )
            if response is None:
                return None
            return response.description

        def select_scene(candidate_scenes: str) -> int | None:
            prompt = SELECT_BEST_SCENE_TEMPLATE.format(scenes=candidate_scenes)
            response = self.generate_text(
                prompt=prompt, temperature=0.5, output_model=SceneIndex
            )
            # Default to first scene if selection fails.
            if response is None:
                return 0
            return response.index

        def refine_scene(scene: str) -> str | None:
            prompt = REFINE_SCENE_TEMPLATE.format(description=scene)
            response = self.generate_text(
                prompt=prompt, temperature=0.7, output_model=SceneDescription
            )
            if response is None:
                return None
            return response.description

        def visualize_scene(scene: str) -> str | None:
            prompt = VISUALIZE_SCENE_TEMPLATE.format(description=scene)
            response = self.generate_text(
                prompt=prompt, temperature=0.7, output_model=SceneDescription
            )
            if response is None:
                return None
            return response.description

        def render_scene(description: str) -> str | None:
            return self.generate_image(description, 1024, 1024, 40)

        graph = GeneratorGraph(
            num_candidates=5,
            original=description,
            generator=generate_scene,
            selector=select_scene,
            refiner=refine_scene,
            visualizer=visualize_scene,
            renderer=render_scene,
        )
        state = await graph.invoke()
        return state.refined, state.image_url

    async def add_action(
        self, scene: str, outcomes: str, name: str, action: str
    ) -> str | None:
        return await self._invoke(
            model=self.text_model,
            prompt=ADD_ACTION_TEMPLATE,
            args={"scene": scene, "outcomes": outcomes, "name": name, "action": action},
        )

    async def end_scene(self, scene: str, outcomes: str) -> str | None:
        return await self._invoke(
            model=self.text_model,
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
