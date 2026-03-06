"""
OmicVerse Jarvis — multi-channel bot bridge for mobile single-cell analysis.

Launch via::

    omicverse jarvis --channel telegram --token BOT_TOKEN

or for Feishu webhook mode::

    omicverse jarvis --channel feishu --feishu-app-id APP_ID --feishu-app-secret APP_SECRET
"""

from .channels.telegram import run_bot
from .channels.feishu import run_feishu_bot

__all__ = ["run_bot", "run_feishu_bot"]
