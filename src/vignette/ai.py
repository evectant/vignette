import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .prompts import (
    ADD_ACTION_TEMPLATE,
    CREATE_SCENE_TEMPLATE,
    END_SCENE_TEMPLATE,
)

logger = logging.getLogger(__name__)


class AI:
    MODEL = "gpt-4o"
    TEMPERATURE = 0.75

    def __init__(self, openai_api_key):
        self.model = ChatOpenAI(
            model=AI.MODEL, temperature=AI.TEMPERATURE, api_key=openai_api_key
        )

    async def create_scene(self, description: str) -> str | None:
        return await self._invoke(
            model=self.model,
            prompt=CREATE_SCENE_TEMPLATE,
            args={"description": description},
        )

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
