"""Optional ntfy.sh notifications (never crash training on failure)."""
from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)
_warned_no_topic = False
_MAX_MESSAGE_LEN = 3500


def notify(
    message: str,
    title: str = "REFiNE",
    priority: str = "default",
    tags: str = "",
    click: str = "",
) -> None:
    """POST a notification to ntfy; no-op if NTFY_TOPIC is unset."""
    global _warned_no_topic

    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        if not _warned_no_topic:
            logger.warning(
                "NTFY_TOPIC not set; notifications disabled (set NTFY_TOPIC to enable)."
            )
            _warned_no_topic = True
        return

    server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    url = f"{server}/{topic}"
    body = message[:_MAX_MESSAGE_LEN]

    headers = {"Title": title, "Priority": priority}
    if tags:
        headers["Tags"] = tags
    if click:
        headers["Click"] = click

    try:
        try:
            import requests

            requests.post(url, data=body.encode("utf-8"), headers=headers, timeout=30)
        except ImportError:
            cmd = ["curl", "-sS", "-X", "POST", url, "-d", body]
            for key, val in headers.items():
                cmd.extend(["-H", f"{key}: {val}"])
            subprocess.run(cmd, check=False, timeout=30)
    except Exception as exc:
        logger.debug("ntfy notify failed: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    notify("ntfy test ok", title="REFiNE test")
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if topic:
        print("sent")
    else:
        print("no-op (NTFY_TOPIC not set)")
