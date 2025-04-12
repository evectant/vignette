import logging
from dataclasses import dataclass

from telegram import Chat, Message, Update, User
from telegram.constants import ParseMode
from telegram.ext import (
    BaseHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .ai import AI

logger = logging.getLogger(__name__)


@dataclass
class Action:
    name: str
    action: str
    outcome: str


@dataclass
class Scene:
    message_id: int
    description: str
    actions: dict[int, Action]

    def outcomes(self) -> str:
        return "\n\n".join(
            [action.outcome for action in self.actions.values() if action.outcome]
        )


class Bot:
    SCENE_KEY = "scene"

    def __init__(self, ai: AI):
        self.ai = ai

    @property
    def handlers(self) -> list[BaseHandler]:
        return [
            CommandHandler("help", self.handle_help),
            CommandHandler("start", self.handle_start),
            CommandHandler("end", self.handle_end),
            CommandHandler("reset", self.handle_reset),
            MessageHandler(filters.REPLY, self.handle_reply),
        ]

    async def handle_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await update.message.set_reaction("👍")
        await update.message.reply_text(
            text="1\. `/start <description>` to create a new scene\. There can be only one active scene\.\n"
            "2\. Reply to the scene message to act\. You may only act once\.\n"
            "3\. `/end` to complete the scene\. Scenes also autocomplete once the majority of chat members reply\.\n"
            "\n"
            "`/reset` to reset the scene\.\n",
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_to_message_id=update.message.message_id,
        )

    async def handle_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not context.args:
            await update.message.reply_text(
                "⚠️ Missing description.",
                reply_to_message_id=update.message.message_id,
            )
            return

        if Bot._is_scene_active(context):
            await update.message.reply_text(
                "⚠️ There is already an active scene.",
                reply_to_message_id=update.message.message_id,
            )
            return

        # Expand the scene description.
        await self._ack_message(update.message)
        initial_description = " ".join(context.args)
        expanded_description, image_url = await self.ai.create_scene(
            initial_description
        )
        if not expanded_description:
            await update.message.reply_text(
                "⚠️ Error while creating the scene.",
                reply_to_message_id=update.message.message_id,
            )
            return

        # Post and store the scene.
        scene_message = await update.message.reply_photo(
            photo=image_url,
            caption=expanded_description,
            reply_to_message_id=update.message.message_id,
        )
        context.chat_data[Bot.SCENE_KEY] = Scene(
            message_id=scene_message.message_id,
            description=expanded_description,
            actions={},
        )

    async def handle_reset(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await update.message.set_reaction("👍")
        Bot._delete_scene(context)

    @staticmethod
    def _normalize_name(user: User) -> str:
        return user.first_name or user.last_name or user.username or str(user.id)

    async def handle_reply(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        scene = Bot._get_scene(context)
        if not scene:
            return

        # Only process replies to the scene message.
        if update.message.reply_to_message.message_id != scene.message_id:
            return

        # Only allow one reply per user.
        if update.message.from_user.id in scene.actions:
            await update.message.reply_text(
                "⚠️ You have already replied to this scene.",
                reply_to_message_id=update.message.message_id,
            )
            return

        # Store right away to prevent the player from replying again while the AI call is in progress.
        user_id = update.message.from_user.id
        action = update.message.text
        name = Bot._normalize_name(update.message.from_user)
        scene.actions[user_id] = Action(
            name=name,
            action=action,
            outcome=None,
        )

        # Generate the outcome.
        await self._ack_message(update.message)
        outcome = await self.ai.add_action(
            scene.description, scene.outcomes(), name, action
        )
        if not outcome:
            await update.message.reply_text(
                "⚠️ Error while processing the action.",
                reply_to_message_id=update.message.message_id,
            )
            # Remove the action, so the player can try again.
            del scene.actions[user_id]
            return

        # Store and post the outcome.
        scene.actions[user_id].outcome = outcome
        await update.message.reply_text(
            outcome, reply_to_message_id=update.message.message_id
        )

        # End scene if we have a majority of replies.
        if await self._has_majority_replied(update.message.chat, context):
            await self.handle_end(update, context)

    async def handle_end(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not Bot._is_scene_active(context):
            await update.message.reply_text(
                "⚠️ No active scene to end.",
                reply_to_message_id=update.message.message_id,
            )
            return

        await self._ack_message(update.message)
        scene = Bot._get_scene(context)
        summary = await self.ai.end_scene(scene.description, scene.outcomes())
        if not summary:
            await update.message.reply_text(
                "⚠️ Error while completing the scene.",
                reply_to_message_id=update.message.message_id,
            )
            return

        await update.message.reply_text(summary, reply_to_message_id=scene.message_id)
        Bot._delete_scene(context)

    async def _ack_message(self, message: Message) -> None:
        await message.set_reaction("👍")
        await message.chat.send_action(action="typing")

    @staticmethod
    def _get_scene(context: ContextTypes.DEFAULT_TYPE) -> Scene | None:
        return context.chat_data.get(Bot.SCENE_KEY, None)

    @staticmethod
    def _delete_scene(context: ContextTypes.DEFAULT_TYPE) -> None:
        if Bot.SCENE_KEY in context.chat_data:
            del context.chat_data[Bot.SCENE_KEY]

    @staticmethod
    def _is_scene_active(context: ContextTypes.DEFAULT_TYPE) -> bool:
        return Bot._get_scene(context) is not None

    @staticmethod
    async def _has_majority_replied(
        chat: Chat, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        scene = Bot._get_scene(context)
        num_replies = len(scene.actions) if scene else 0
        num_members = await chat.get_member_count()
        majority_replied = num_replies > num_members // 2
        logger.info(f"{num_members=}, {num_replies=}, {majority_replied=}")
        return majority_replied
