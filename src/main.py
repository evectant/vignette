import logging
import os
import sys

from dotenv import load_dotenv
from rich.logging import RichHandler
from telegram.ext import Application

from vignette.ai import AI
from vignette.bot import Bot


def configure_logging():
    logging.basicConfig(
        format="%(asctime)s - %(name)s:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[RichHandler()],
    )


configure_logging()
logger = logging.getLogger(__name__)


def getenv(key: str) -> str:
    value = os.getenv(key)
    if not value:
        message = f"{key} is missing or empty in the environment"
        logger.error(message)
        sys.exit(message)
    return value


def main():
    logger.info(f"Starting with {sys.executable=} and {sys.path=}")

    load_dotenv()
    llm_api_key = getenv("ANTHROPIC_API_KEY")
    runware_api_key = getenv("RUNWARE_API_KEY")
    telegram_api_key = getenv("TELEGRAM_API_KEY")

    ai = AI(llm_api_key, runware_api_key)
    bot = Bot(ai)

    app = Application.builder().token(telegram_api_key).build()
    app.add_handlers(bot.handlers)
    app.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
