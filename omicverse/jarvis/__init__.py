"""
OmicVerse Jarvis — multi-channel bot bridge for mobile single-cell analysis.

Launch via::

    omicverse jarvis --channel telegram --token BOT_TOKEN

or for Feishu mode::

    omicverse jarvis --channel feishu --feishu-app-id APP_ID --feishu-app-secret APP_SECRET --feishu-connection-mode websocket

or for iMessage mode::

    omicverse jarvis --channel imessage --imessage-cli-path imsg
"""

from .channels.telegram import run_bot
from .channels.feishu import run_feishu_bot, run_feishu_ws_bot
from .channels.imessage import run_imessage_bot

__all__ = ["run_bot", "run_feishu_bot", "run_feishu_ws_bot", "run_imessage_bot"]
