import pytest

from omicverse.jarvis.channels.telegram import TelegramDelivery


class _BotNotModified:
    def __init__(self) -> None:
        self.calls = 0

    async def edit_message_text(self, **kwargs):
        self.calls += 1
        raise Exception("BadRequest: Message is not modified")


@pytest.mark.asyncio
async def test_edit_text_treats_not_modified_as_success() -> None:
    delivery = TelegramDelivery(
        bot=_BotNotModified(),
        chat_lock_factory=lambda _chat_id: None,
        keyboard_factory=lambda _controls: None,
    )

    ok = await delivery._edit_text(
        123,
        456,
        "<b>same</b>",
        parse_mode="HTML",
        reply_markup=None,
    )

    assert ok is True
