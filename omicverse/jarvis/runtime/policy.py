from __future__ import annotations

import re

from .models import MessageEnvelope, PolicyDecision, RuntimeTaskState


class MessagePolicy:
    """Decide whether an incoming message starts work or joins the queue."""

    def decide(self, envelope: MessageEnvelope, state: RuntimeTaskState) -> PolicyDecision:
        if self._requires_explicit_trigger(envelope) and not envelope.explicit_trigger:
            return PolicyDecision(
                action="ignore",
                should_ignore=True,
                reason="explicit_trigger_required",
            )

        if state.running:
            if envelope.route.is_direct:
                return PolicyDecision(
                    action="queue",
                    should_queue=True,
                    should_quick_chat=True,
                    reason="analysis_running",
                )
            if self._looks_like_analysis_request(envelope):
                return PolicyDecision(
                    action="queue",
                    should_queue=True,
                    reason="analysis_running_queue",
                )
            return PolicyDecision(
                action="quick_chat",
                should_quick_chat=True,
                reason="analysis_running_followup",
            )

        return PolicyDecision(
            action="start",
            should_ack=True,
            should_start=True,
            reason="idle",
        )

    @staticmethod
    def _requires_explicit_trigger(envelope: MessageEnvelope) -> bool:
        return not envelope.route.is_direct

    @classmethod
    def _looks_like_analysis_request(cls, envelope: MessageEnvelope) -> bool:
        override = envelope.metadata.get("analysis_like")
        if isinstance(override, bool):
            return override

        text = (envelope.text or "").strip().lower()
        if not text:
            return False
        if envelope.trigger == "command":
            return True
        if len(text) >= 140 or "\n" in text:
            return True
        if "```" in text or "`" in text:
            return True
        if re.search(
            r"\b("
            r"analy[sz]e|analysis|run|plot|draw|generate|create|compute|calculate|"
            r"cluster|umap|pca|qc|filter|subset|find|identify|compare|marker|"
            r"differential|de\\b|trajectory|annotate|export|save|load"
            r")\b",
            text,
        ):
            return True
        if text.endswith("?") and len(text.split()) <= 18:
            return False
        return len(text.split()) >= 18
