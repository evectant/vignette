import logging
import uuid
from typing import Type, TypeVar

import requests
from langchain.chat_models import init_chat_model
from pydantic import BaseModel
import tenacity

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
    TEXT_MODEL = "anthropic:claude-3-7-sonnet-latest"
    IMAGE_MODEL = "runware:101@1"  # FLUX.1 Dev.

    def __init__(self, llm_api_key: str, runware_api_key: str):
        self.text_model = init_chat_model(
            model=AI.TEXT_MODEL,
            api_key=llm_api_key,
            configurable_fields=("temperature"),
        )
        self.runware_api_key = runware_api_key

    retry = tenacity.retry(
        stop=tenacity.stop_after_attempt(4),
        wait=tenacity.wait_exponential(),
        reraise=True,
        after=lambda state: logger.error(
            f"Attempt {state.attempt_number} failed with {state.outcome.exception()}"
        ),
    )

    def generate_text(
        self, prompt: str, temperature: float, output_model: Type[T]
    ) -> T:
        logger.info(
            f"Invoking {self.text_model=} with {prompt=}, {temperature=}, {output_model=}"
        )

        @AI.retry
        def make_request():
            return self.text_model.with_structured_output(output_model).invoke(
                prompt, config={"configurable": {"temperature": temperature}}
            )

        response = make_request()
        logger.info(f"Received {response=}")
        return response

    def generate_image(self, prompt: str) -> str:
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
                "width": 1024,
                "height": 1024,
                "steps": 50,
                "outputFormat": "PNG",
                # Prompt adherence. FLUX.1 Dev generates heavily stylized images at 0.0
                # and realistic images at 2.0 and higher. 1.4 strikes a good balance.
                "CFGScale": 1.4,
                "numberResults": 1,
                "includeCost": True,
            }
        ]

        @AI.retry
        def make_request():
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response

        response = make_request()
        response_json = response.json()
        logger.info(f"Received {response_json=} in response to {data=}")

        return response_json["data"][0]["imageURL"]

    async def create_scene(self, description: str) -> tuple[str, str] | None:
        class SceneDescription(BaseModel):
            description: str

        class SceneIndex(BaseModel):
            index: int

        def generate_scene(inputs: list[str]) -> str | None:
            prompt = CREATE_SCENE_TEMPLATE.format(description=inputs[0])
            response = self.generate_text(
                prompt=prompt, temperature=0.8, output_model=SceneDescription
            )
            return response.description

        def select_scene(candidate_scenes: str) -> int | None:
            prompt = SELECT_BEST_SCENE_TEMPLATE.format(scenes=candidate_scenes)
            response = self.generate_text(
                prompt=prompt, temperature=0.5, output_model=SceneIndex
            )
            return response.index

        def refine_scene(scene: str) -> str | None:
            prompt = REFINE_SCENE_TEMPLATE.format(description=scene)
            response = self.generate_text(
                prompt=prompt, temperature=0.7, output_model=SceneDescription
            )
            return response.description

        def visualize_scene(scene: str) -> str | None:
            prompt = VISUALIZE_SCENE_TEMPLATE.format(description=scene)
            response = self.generate_text(
                prompt=prompt, temperature=0.7, output_model=SceneDescription
            )
            return response.description

        def render_scene(description: str) -> str | None:
            return self.generate_image(description)

        graph = GeneratorGraph(
            num_candidates=3,
            inputs=[description],
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
        class Outcome(BaseModel):
            description: str

        def generate_outcome(inputs: list[str]) -> str | None:
            prompt = ADD_ACTION_TEMPLATE.format(
                scene=inputs[0], outcomes=inputs[1], name=inputs[2], action=inputs[3]
            )
            response = self.generate_text(
                prompt=prompt, temperature=0.8, output_model=Outcome
            )
            return response.description

        def refine_outcome(outcome: str) -> str | None:
            prompt = REFINE_SCENE_TEMPLATE.format(description=outcome)
            response = self.generate_text(
                prompt=prompt, temperature=0.7, output_model=Outcome
            )
            return response.description

        graph = GeneratorGraph(
            num_candidates=1,
            inputs=[scene, outcomes, name, action],
            generator=generate_outcome,
            selector=lambda _: 0,  # There is only one candidate.
            refiner=refine_outcome,
            visualizer=lambda _: None,
            renderer=lambda _: None,
        )
        state = await graph.invoke()
        return state.refined

    async def end_scene(self, scene: str, outcomes: str) -> str | None:
        class Ending(BaseModel):
            description: str

        class EndingIndex(BaseModel):
            index: int

        def generate_ending(inputs: list[str]) -> str | None:
            prompt = END_SCENE_TEMPLATE.format(scene=inputs[0], outcomes=inputs[1])
            response = self.generate_text(
                prompt=prompt, temperature=0.8, output_model=Ending
            )
            return response.description

        def select_ending(candidate_endings: str) -> int | None:
            prompt = SELECT_BEST_SCENE_TEMPLATE.format(scenes=candidate_endings)
            response = self.generate_text(
                prompt=prompt, temperature=0.5, output_model=EndingIndex
            )
            return response.index

        def refine_ending(ending: str) -> str | None:
            prompt = REFINE_SCENE_TEMPLATE.format(description=ending)
            response = self.generate_text(
                prompt=prompt, temperature=0.7, output_model=Ending
            )
            return response.description

        graph = GeneratorGraph(
            num_candidates=3,
            inputs=[scene, outcomes],
            generator=generate_ending,
            selector=select_ending,
            refiner=refine_ending,
            # It would be nice to visualize the ending, but it needs
            # to be consistent with the scene, which requires some thought.
            visualizer=lambda _: None,
            renderer=lambda _: None,
        )
        state = await graph.invoke()
        return state.refined
