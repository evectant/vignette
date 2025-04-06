import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from vignette.ai import AI

MODEL_OUTPUT = "Model-generated text"


@pytest.fixture
def mock_ai(mocker):
    ai = AI(openai_api_key="mock-key")
    ai.model = mocker.MagicMock(spec=BaseChatModel)
    ai.model.invoke = mocker.MagicMock(
        return_value=mocker.MagicMock(content=MODEL_OUTPUT)
    )
    ai.model.with_structured_output.return_value = mocker.MagicMock(
        invoke=mocker.MagicMock(return_value=mocker.MagicMock(description=MODEL_OUTPUT))
    )
    return ai


@pytest.fixture
def mock_working_ai(mocker, mock_ai):
    return mock_ai


@pytest.fixture
def mock_failing_ai(mocker, mock_ai):
    mock_ai.model.invoke = mocker.MagicMock(side_effect=Exception("API error"))
    mock_ai.model.with_structured_output.return_value = mocker.MagicMock(
        invoke=mocker.MagicMock(side_effect=Exception("API error"))
    )
    return mock_ai


@pytest.mark.asyncio
async def test_creates_scene(mock_working_ai):
    result = await mock_working_ai.create_scene("Initial scene")
    assert result == MODEL_OUTPUT
    assert mock_working_ai.model.with_structured_output.call_count == 6


@pytest.mark.asyncio
async def test_handles_errors_when_creating_scene(mock_failing_ai):
    result = await mock_failing_ai.create_scene("Initial scene")
    assert result is None
    assert mock_failing_ai.model.with_structured_output.call_count == 6


@pytest.mark.asyncio
async def test_adds_action(mock_working_ai):
    result = await mock_working_ai.add_action("Scene", "Outcomes", "Name", "Action")
    assert result == MODEL_OUTPUT
    mock_working_ai.model.invoke.assert_called_once()


@pytest.mark.asyncio
async def test_handles_errors_when_adding_action(mock_failing_ai):
    result = await mock_failing_ai.add_action("Scene", "Outcomes", "Name", "Action")
    assert result is None
    mock_failing_ai.model.invoke.assert_called_once()


@pytest.mark.asyncio
async def test_ends_scene(mock_working_ai):
    result = await mock_working_ai.end_scene("Scene", "Outcomes")
    assert result == MODEL_OUTPUT
    mock_working_ai.model.invoke.assert_called_once()


@pytest.mark.asyncio
async def test_handles_errors_when_ending_scene(mock_failing_ai):
    result = await mock_failing_ai.end_scene("Scene", "Outcomes")
    assert result is None
    mock_failing_ai.model.invoke.assert_called_once()
