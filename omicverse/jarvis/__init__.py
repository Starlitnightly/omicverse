"""
OmicVerse Jarvis — Telegram bot bridge for mobile single-cell analysis.

Launch via::

    omicverse jarvis --token BOT_TOKEN

or set ``TELEGRAM_BOT_TOKEN`` env var and run::

    omicverse jarvis
"""

from .bot import run_bot

__all__ = ["run_bot"]
