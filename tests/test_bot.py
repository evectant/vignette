import pytest
from telegram import Chat, Message, Update, User
from telegram.ext import ContextTypes

from vignette.bot import Bot, Scene

SCENE_DESCRIPTION = "Expanded scene"
SCENE_IMAGE_URL = "https://example.com/image.png"
ACTION_INTENT = "Action intent"
ACTION_OUTCOME = "Action outcome"
SCENE_SUMMARY = "Scene summary"

USER_COMMAND_MESSAGE_ID = 1001
BOT_COMMAND_RESPONSE_MESSAGE_ID = 1002
BOT_SCENE_MESSAGE_ID = 1003
USER_ACTION_MESSAGE_ID = 1004
BOT_ACTION_RESPONSE_MESSAGE_ID = 1005
BOT_SCENE_SUMMARY_MESSAGE_ID = 1006

USER_ID = 2001
USER_USERNAME = "username"
USER_FIRST_NAME = "First"
USER_LAST_NAME = "Last"

NUM_MEMBERS = 5


@pytest.fixture
def update(mocker):
    mock_update = mocker.MagicMock(spec=Update)

    mock_update.message = mocker.MagicMock(spec=Message)
    mock_update.message.chat = mocker.MagicMock(spec=Chat)
    mock_update.message.chat.send_action = mocker.AsyncMock()
    mock_update.message.reply_text = mocker.AsyncMock()
    mock_update.message.reply_photo = mocker.AsyncMock()
    mock_update.message.set_reaction = mocker.AsyncMock()
    mock_update.effective_chat = mock_update.message.chat
    mock_update.effective_chat.get_member_count = mocker.AsyncMock(
        return_value=NUM_MEMBERS
    )

    mock_update.message.from_user = mocker.MagicMock(spec=User)
    mock_update.message.from_user.id = USER_ID
    mock_update.message.from_user.username = USER_USERNAME
    mock_update.message.from_user.first_name = USER_FIRST_NAME
    mock_update.message.from_user.last_name = USER_LAST_NAME

    # Model all player messages as replies to the scene message.
    mock_update.message.message_id = USER_ACTION_MESSAGE_ID
    mock_update.message.reply_to_message = mocker.MagicMock(
        spec=Message, message_id=BOT_SCENE_MESSAGE_ID
    )
    mock_update.message.text = ACTION_INTENT

    return mock_update


@pytest.fixture
def context(mocker):
    context = mocker.MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    context.args = ["Initial", "scene"]
    context.chat_data = {}
    return context


@pytest.fixture
def bot(mocker):
    mock_ai = mocker.MagicMock()
    mock_ai.create_scene = mocker.AsyncMock(
        return_value=(SCENE_DESCRIPTION, SCENE_IMAGE_URL)
    )
    mock_ai.add_action = mocker.AsyncMock(return_value=ACTION_OUTCOME)
    mock_ai.end_scene = mocker.AsyncMock(return_value=SCENE_SUMMARY)
    return Bot(ai=mock_ai)


def called_with_warning(mock_callable):
    call_args, call_kwargs = mock_callable.call_args
    if call_args:
        text = call_args[0]
    elif call_kwargs and "text" in call_kwargs:
        text = call_kwargs["text"]
    elif call_kwargs and "caption" in call_kwargs:
        text = call_kwargs["caption"]
    else:
        raise ValueError("No text or caption found in call")
    return "⚠️" in text


@pytest.mark.asyncio
async def test_posts_help(bot, update, context):
    await bot.handle_help(update, context)

    # Verify that help was posted
    update.message.reply_text.assert_called_once()
    # ...with expected content
    _, call_kwargs = update.message.reply_text.call_args
    assert "/start" in call_kwargs.get("text")
    # ...as a reply to the original message.
    assert call_kwargs.get("reply_to_message_id") == update.message.message_id


@pytest.mark.asyncio
async def test_creates_scene(bot, update, context, mocker):
    assert not Bot._is_scene_active(context)
    update.message.reply_photo.return_value = mocker.MagicMock(
        message_id=BOT_SCENE_MESSAGE_ID
    )
    await bot.handle_start(update, context)

    # Verify that AI was called
    bot.ai.create_scene.assert_called_once_with("Initial scene")
    # ...the result was posted as a reply to the original message
    update.message.reply_photo.assert_called_once_with(
        photo=SCENE_IMAGE_URL,
        caption=SCENE_DESCRIPTION,
        reply_to_message_id=update.message.message_id,
    )
    # ...and the scene was stored in the context.
    assert Bot._is_scene_active(context)
    scene = Bot._get_scene(context)
    assert isinstance(scene, Scene)
    assert scene.message_id == BOT_SCENE_MESSAGE_ID
    assert scene.description == SCENE_DESCRIPTION
    assert scene.actions == {}


@pytest.mark.asyncio
async def test_handles_empty_scene_description(bot, update, context):
    assert not Bot._is_scene_active(context)
    context.args = []
    await bot.handle_start(update, context)

    # Verify that no scene was created
    assert not Bot._is_scene_active(context)
    # ...no AI was called
    bot.ai.create_scene.assert_not_called()
    # ...and a reply was posted
    update.message.reply_text.assert_called_once()
    # ...that contained a warning.
    assert called_with_warning(update.message.reply_text)


@pytest.mark.asyncio
async def test_does_not_overwrite_existing_scene(bot, update, context):
    assert not Bot._is_scene_active(context)
    await bot.handle_start(update, context)

    # Verify that a scene was created
    assert Bot._is_scene_active(context)
    scene_id = id(Bot._get_scene(context))
    # ...AI was called
    assert bot.ai.create_scene.call_count == 1
    # ...and a normal reply was posted.
    assert update.message.reply_photo.call_count == 1
    assert not called_with_warning(update.message.reply_photo)

    # Try to create another scene.
    await bot.handle_start(update, context)

    # Verify that the scene remained unchanged
    assert id(Bot._get_scene(context)) == scene_id
    # ...there were no new AI calls
    assert bot.ai.create_scene.call_count == 1
    # ...but another reply was posted
    assert update.message.reply_text.call_count == 1
    # ...that contained a warning.
    assert called_with_warning(update.message.reply_text)


@pytest.mark.asyncio
async def test_handles_ai_errors_when_creating_scene(bot, update, context):
    assert not Bot._is_scene_active(context)
    bot.ai.create_scene.return_value = None, None
    await bot.handle_start(update, context)

    # Verify that AI was called
    bot.ai.create_scene.assert_called_once()
    # ...but no scene was created
    assert not Bot._is_scene_active(context)
    # ...and a reply was posted
    update.message.reply_text.assert_called_once()
    # ...that contained a warning.
    assert called_with_warning(update.message.reply_text)


@pytest.mark.asyncio
async def test_resets_scene(bot, update, context):
    # Verify that we can reset even when no scene exists.
    assert not Bot._is_scene_active(context)
    await bot.handle_reset(update, context)
    assert not Bot._is_scene_active(context)

    # Create a scene
    await bot.handle_start(update, context)
    assert Bot._is_scene_active(context)
    # ...reset
    await bot.handle_reset(update, context)
    assert not Bot._is_scene_active(context)
    # ...and create again.
    await bot.handle_start(update, context)
    assert Bot._is_scene_active(context)


@pytest.mark.asyncio
async def test_handles_replies(bot, update, context, mocker):
    # Set up a scene and add a reply.
    context.chat_data[Bot.SCENE_KEY] = Scene(
        message_id=BOT_SCENE_MESSAGE_ID, description=SCENE_DESCRIPTION, actions={}
    )
    await bot.handle_reply(update, context)

    # Verify that AI was called
    assert bot.ai.add_action.call_count == 1
    # ...a normal reply was posted
    assert update.message.reply_text.call_count == 1
    assert not called_with_warning(update.message.reply_text)
    # ...and the action was stored in the scene.
    scene = Bot._get_scene(context)
    assert USER_ID in scene.actions
    assert scene.actions[USER_ID].name == USER_FIRST_NAME
    assert scene.actions[USER_ID].action == ACTION_INTENT
    assert scene.actions[USER_ID].outcome == ACTION_OUTCOME

    # Add a second reply from another user.
    update.message.from_user.id = USER_ID + 1
    update.message.from_user.first_name = USER_FIRST_NAME + "a"
    await bot.handle_reply(update, context)

    # Verify that AI was called
    assert bot.ai.add_action.call_count == 2
    # ...a normal reply was posted
    assert update.message.reply_text.call_count == 2
    assert not called_with_warning(update.message.reply_text)
    # ...and the action was stored in the scene.
    scene = Bot._get_scene(context)
    assert len(scene.actions) == 2
    assert USER_ID + 1 in scene.actions
    assert scene.actions[USER_ID + 1].name == USER_FIRST_NAME + "a"
    assert scene.actions[USER_ID + 1].action == ACTION_INTENT
    assert scene.actions[USER_ID + 1].outcome == ACTION_OUTCOME

    # Add a third reply from another user, triggering scene completion.
    update.message.from_user.id = USER_ID + 2
    update.message.from_user.first_name = USER_FIRST_NAME + "b"
    await bot.handle_reply(update, context)

    # Verify that AI was called
    assert bot.ai.add_action.call_count == 3
    # ...a normal reply and scene summary were posted
    assert update.message.reply_text.call_count == 4
    assert not called_with_warning(update.message.reply_text)
    # ...and the scene was deleted.
    assert not Bot._is_scene_active(context)


@pytest.mark.asyncio
async def test_skips_replies_without_active_scene(bot, update, context):
    assert not Bot._is_scene_active(context)
    await bot.handle_reply(update, context)

    # Verify that no scene was created
    assert not Bot._is_scene_active(context)
    # ...no AI was called
    bot.ai.add_action.assert_not_called()
    # ...and no reply was posted.
    update.message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_skips_replies_to_non_scene_messages(bot, update, context):
    # Set up a scene, but reply to a different message.
    context.chat_data[Bot.SCENE_KEY] = Scene(
        message_id=BOT_SCENE_MESSAGE_ID, description=SCENE_DESCRIPTION, actions={}
    )
    update.message.reply_to_message.message_id = BOT_SCENE_MESSAGE_ID + 1
    await bot.handle_reply(update, context)

    # Verify that no actions were added
    assert not Bot._get_scene(context).actions
    # ...no AI was called
    bot.ai.add_action.assert_not_called()
    # ...and no reply was posted.
    update.message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_allows_only_one_reply_per_user(bot, update, context):
    # Set up a scene and add a reply.
    context.chat_data[Bot.SCENE_KEY] = Scene(
        message_id=BOT_SCENE_MESSAGE_ID, description=SCENE_DESCRIPTION, actions={}
    )
    update.message.reply_to_message.message_id = BOT_SCENE_MESSAGE_ID
    await bot.handle_reply(update, context)

    # Verify that one action was added
    assert len(Bot._get_scene(context).actions) == 1
    # ...AI was called
    bot.ai.add_action.assert_called_once()
    # ...and a normal reply was posted.
    update.message.reply_text.assert_called_once()
    assert not called_with_warning(update.message.reply_text)

    # Try to add another reply from the same user.
    await bot.handle_reply(update, context)

    # Verify that no new action was added
    assert len(Bot._get_scene(context).actions) == 1
    # ...no new AI call was made
    bot.ai.add_action.assert_called_once()
    # ...and a reply with a warning was posted.
    assert update.message.reply_text.call_count == 2
    assert called_with_warning(update.message.reply_text)


@pytest.mark.asyncio
async def test_handles_ai_errors_when_adding_actions(bot, update, context):
    # Set up a scene and add a reply but make AI fail.
    context.chat_data[Bot.SCENE_KEY] = Scene(
        message_id=BOT_SCENE_MESSAGE_ID, description=SCENE_DESCRIPTION, actions={}
    )
    bot.ai.add_action.return_value = None
    await bot.handle_reply(update, context)

    # Verify that AI was called
    bot.ai.add_action.assert_called_once()
    # ...a reply with a warning was posted
    assert update.message.reply_text.call_count == 1
    assert called_with_warning(update.message.reply_text)
    # ...and the in-progress action was removed from the scene.
    scene = Bot._get_scene(context)
    assert USER_ID not in scene.actions


@pytest.mark.asyncio
async def test_completes_scene(bot, update, context):
    # Set up a scene and complete it.
    context.chat_data[Bot.SCENE_KEY] = Scene(
        message_id=BOT_SCENE_MESSAGE_ID, description=SCENE_DESCRIPTION, actions={}
    )
    assert Bot._is_scene_active(context)
    await bot.handle_end(update, context)

    # Verify that it was completed
    assert not Bot._is_scene_active(context)
    # ...a scene summary was posted.
    update.message.reply_text.assert_called_once()
    assert not called_with_warning(update.message.reply_text)


@pytest.mark.asyncio
async def test_handles_ai_errors_when_completing_scenes(bot, update, context):
    # Set up a scene...
    context.chat_data[Bot.SCENE_KEY] = Scene(
        message_id=BOT_SCENE_MESSAGE_ID, description=SCENE_DESCRIPTION, actions={}
    )
    # ...and complete it, but make AI fail.
    bot.ai.end_scene.return_value = None
    await bot.handle_end(update, context)

    # Verify that AI was called
    bot.ai.end_scene.assert_called_once()
    # ...a reply with a warning was posted
    update.message.reply_text.assert_called_once()
    assert called_with_warning(update.message.reply_text)
    # ...and the scene remained active.
    assert Bot._is_scene_active(context)
