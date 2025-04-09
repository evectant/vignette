import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from vignette.ai import AI

TEXT_MODEL_OUTPUT = "Model-generated text"
IMAGE_MODEL_OUTPUT = "https://example.com/image.png"


@pytest.fixture
def mock_ai(mocker):
    ai = AI(openai_api_key="mock-openai-key", runware_api_key="mock-runware-key")
    ai.text_model = mocker.MagicMock(spec=BaseChatModel)
    ai.text_model.invoke = mocker.MagicMock(
        return_value=mocker.MagicMock(content=TEXT_MODEL_OUTPUT)
    )
    ai.text_model.with_structured_output.return_value = mocker.MagicMock(
        invoke=mocker.MagicMock(
            return_value=mocker.MagicMock(description=TEXT_MODEL_OUTPUT)
        )
    )
    ai.image_model = mocker.MagicMock()
    ai.generate_image = mocker.MagicMock(return_value=IMAGE_MODEL_OUTPUT)
    return ai


@pytest.fixture
def mock_working_ai(mocker, mock_ai):
    return mock_ai


@pytest.fixture
def mock_failing_ai(mocker, mock_ai):
    mock_ai.text_model.invoke = mocker.MagicMock(side_effect=Exception("API error"))
    mock_ai.text_model.with_structured_output.return_value = mocker.MagicMock(
        invoke=mocker.MagicMock(side_effect=Exception("API error"))
    )
    # TODO: A better mock would raise an exception, but we first need to change
    # how we handle exceptions (only catch at the bot command level).
    mock_ai.generate_image = mocker.MagicMock(return_value=None)
    return mock_ai


@pytest.mark.asyncio
async def test_creates_scene(mock_working_ai):
    text, image = await mock_working_ai.create_scene("Initial scene")
    assert text == TEXT_MODEL_OUTPUT
    assert image == IMAGE_MODEL_OUTPUT
    # 6 calls: 3 to generate candidates, 1 to select, 1 to refine, 1 to visualize.
    assert mock_working_ai.text_model.with_structured_output.call_count == 6
    assert mock_working_ai.generate_image.call_count == 1


@pytest.mark.asyncio
async def test_handles_errors_when_creating_scene(mock_failing_ai):
    text, image = await mock_failing_ai.create_scene("Initial scene")
    assert text is None
    assert image is None
    assert mock_failing_ai.text_model.with_structured_output.call_count == 6
    assert mock_failing_ai.generate_image.call_count == 1


@pytest.mark.asyncio
async def test_adds_action(mock_working_ai):
    result = await mock_working_ai.add_action("Scene", "Outcomes", "Name", "Action")
    assert result == TEXT_MODEL_OUTPUT
    mock_working_ai.text_model.invoke.assert_called_once()


@pytest.mark.asyncio
async def test_handles_errors_when_adding_action(mock_failing_ai):
    result = await mock_failing_ai.add_action("Scene", "Outcomes", "Name", "Action")
    assert result is None
    mock_failing_ai.text_model.invoke.assert_called_once()


@pytest.mark.asyncio
async def test_ends_scene(mock_working_ai):
    result = await mock_working_ai.end_scene("Scene", "Outcomes")
    assert result == TEXT_MODEL_OUTPUT
    # 5 calls: 3 to generate candidates, 1 to select, 1 to refine (no visualization).
    assert mock_working_ai.text_model.with_structured_output.call_count == 5


@pytest.mark.asyncio
async def test_handles_errors_when_ending_scene(mock_failing_ai):
    result = await mock_failing_ai.end_scene("Scene", "Outcomes")
    assert result is None
    assert mock_failing_ai.text_model.with_structured_output.call_count == 5
