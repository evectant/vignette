import pytest
import tenacity
from langchain_core.language_models.chat_models import BaseChatModel

from vignette.ai import AI

TEXT_MODEL_OUTPUT = "Model-generated text"
IMAGE_MODEL_OUTPUT = "https://example.com/image.png"

TEXT_MODEL_ERROR = "Text model error"
IMAGE_MODEL_ERROR = "Image model error"


@pytest.fixture
def mock_ai(mocker):
    ai = AI(llm_api_key="mock-llm-key", runware_api_key="mock-runware-key")
    ai.text_model = mocker.MagicMock(spec=BaseChatModel)
    ai.text_model.invoke = mocker.MagicMock(
        return_value=mocker.MagicMock(content=TEXT_MODEL_OUTPUT)
    )
    ai.text_model.with_structured_output.return_value = mocker.MagicMock(
        invoke=mocker.MagicMock(
            return_value=mocker.MagicMock(description=TEXT_MODEL_OUTPUT)
        )
    )
    ai.generate_image = mocker.MagicMock(return_value=IMAGE_MODEL_OUTPUT)

    AI.retry = tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_fixed(0.01),
        reraise=True,
    )

    return ai


@pytest.fixture
def mock_working_ai(mocker, mock_ai):
    return mock_ai


@pytest.fixture
def mock_failing_ai(mocker, mock_ai):
    mock_ai.text_model.with_structured_output.return_value = mocker.MagicMock(
        invoke=mocker.MagicMock(side_effect=Exception(TEXT_MODEL_ERROR))
    )
    mock_ai.generate_image = mocker.MagicMock(side_effect=Exception(IMAGE_MODEL_ERROR))
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
    with pytest.raises(Exception) as ex:
        await mock_failing_ai.create_scene("Initial scene")
    assert str(ex.value) == TEXT_MODEL_ERROR
    # We can't assert the exact number of calls because the final exception is raised once any
    # graph node exhausts its retries, and we don't know how many nodes still have retries left.
    # We could assert a tighter bound, but not worth it for now.
    assert mock_failing_ai.text_model.with_structured_output.call_count > 0
    # The graph should not progress to image generation.
    assert mock_failing_ai.generate_image.not_called()


@pytest.mark.asyncio
async def test_adds_action(mock_working_ai):
    result = await mock_working_ai.add_action("Scene", "Outcomes", "Name", "Action")
    assert result == TEXT_MODEL_OUTPUT
    assert mock_working_ai.text_model.with_structured_output.call_count == 2


@pytest.mark.asyncio
async def test_handles_errors_when_adding_action(mock_failing_ai):
    with pytest.raises(Exception) as ex:
        await mock_failing_ai.add_action("Scene", "Outcomes", "Name", "Action")
    assert str(ex.value) == TEXT_MODEL_ERROR
    assert mock_failing_ai.text_model.with_structured_output.call_count > 0


@pytest.mark.asyncio
async def test_ends_scene(mock_working_ai):
    result = await mock_working_ai.end_scene("Scene", "Outcomes")
    assert result == TEXT_MODEL_OUTPUT
    # 5 calls: 3 to generate candidates, 1 to select, 1 to refine (no visualization).
    assert mock_working_ai.text_model.with_structured_output.call_count == 5


@pytest.mark.asyncio
async def test_handles_errors_when_ending_scene(mock_failing_ai):
    with pytest.raises(Exception) as ex:
        await mock_failing_ai.end_scene("Scene", "Outcomes")
    assert str(ex.value) == TEXT_MODEL_ERROR
    assert mock_failing_ai.text_model.with_structured_output.call_count > 0
