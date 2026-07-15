#!/usr/bin/env python3
"""Post the currently-due tweet from the AgentFarm launch schedule.

Reads ``docs/communications/tweet_schedule.json``, determines which tweet is
scheduled for the current UTC time, and posts it to X (Twitter) via the v2
``POST /2/tweets`` endpoint using OAuth 1.0a user-context authentication.

Designed to be triggered by a scheduler (e.g. a Cursor Automation with a cron
trigger of ``0 9,13,18 * * *``). Each invocation posts at most one tweet: the
one whose scheduled time falls within the lateness window before "now". This
keeps the script stateless — no posted-tweet ledger is needed as long as the
scheduler fires once per slot.

Credentials are read from environment variables (see ``OAuthCredentials``).
Uses only the Python standard library so it runs without the project venv.

Usage:
    python scripts/post_scheduled_tweet.py                # post the due tweet
    python scripts/post_scheduled_tweet.py --dry-run      # print, don't post
    python scripts/post_scheduled_tweet.py --at 2026-07-20T09:00:00  # test a slot
    python scripts/post_scheduled_tweet.py --list         # show the full schedule

Exit codes: 0 = posted (or dry-run), 2 = no tweet due, 1 = error.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import secrets as _secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEDULE_PATH = REPO_ROOT / "docs" / "communications" / "tweet_schedule.json"
POST_TWEET_URL = "https://api.x.com/2/tweets"

#: A tweet is considered "due" from its scheduled time until this much later.
#: Slots are at least 4 hours apart, so a 2-hour window can never match two.
DEFAULT_LATENESS = timedelta(hours=2)

#: X counts any URL as a fixed t.co length regardless of the actual URL.
TCO_URL_LENGTH = 23
MAX_WEIGHTED_LENGTH = 280

#: Unicode ranges that weigh 1 unit in X's character counting; everything
#: else (CJK, emoji, ...) weighs 2. Mirrors the twitter-text v3 config.
_LIGHT_WEIGHT_RANGES = ((0, 4351), (8192, 8205), (8208, 8223), (8242, 8247))


@dataclass(frozen=True)
class ScheduledTweet:
    day: int
    slot: str
    text: str
    scheduled_at: datetime


@dataclass(frozen=True)
class OAuthCredentials:
    """OAuth 1.0a user-context credentials for the account being posted to."""

    api_key: str
    api_secret: str
    access_token: str
    access_token_secret: str

    ENV_VARS = ("X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET")

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "OAuthCredentials":
        missing = [name for name in cls.ENV_VARS if not env.get(name)]
        if missing:
            raise KeyError(f"Missing required environment variables: {', '.join(missing)}")
        return cls(*(env[name] for name in cls.ENV_VARS))


def load_schedule(path: Path = DEFAULT_SCHEDULE_PATH) -> list[ScheduledTweet]:
    """Load and expand the schedule JSON into concrete scheduled datetimes."""
    data = json.loads(path.read_text(encoding="utf-8"))
    campaign = data["campaign"]
    start = datetime.fromisoformat(campaign["start_date"]).replace(tzinfo=timezone.utc)
    slot_hours: dict[str, int] = campaign["slot_hours"]

    tweets = []
    for entry in data["tweets"]:
        scheduled_at = start + timedelta(days=entry["day"] - 1, hours=slot_hours[entry["slot"]])
        tweets.append(
            ScheduledTweet(day=entry["day"], slot=entry["slot"], text=entry["text"], scheduled_at=scheduled_at)
        )
    return sorted(tweets, key=lambda t: t.scheduled_at)


def resolve_due_tweet(
    tweets: Sequence[ScheduledTweet],
    now: datetime,
    lateness: timedelta = DEFAULT_LATENESS,
) -> Optional[ScheduledTweet]:
    """Return the tweet whose slot covers ``now``, or None if nothing is due."""
    for tweet in tweets:
        if tweet.scheduled_at <= now < tweet.scheduled_at + lateness:
            return tweet
    return None


def weighted_tweet_length(text: str) -> int:
    """Estimate X's weighted character count for ``text``.

    URLs count as a fixed 23 units; codepoints in the light-weight ranges
    count 1, all others 2. Multi-codepoint emoji are overcounted slightly,
    which errs on the safe side of the 280 limit.
    """
    total = 0
    for token in text.split(" "):
        if token.startswith(("http://", "https://")):
            total += TCO_URL_LENGTH
        else:
            for char in token:
                cp = ord(char)
                total += 1 if any(lo <= cp <= hi for lo, hi in _LIGHT_WEIGHT_RANGES) else 2
    # Add back the separating spaces consumed by split().
    total += text.count(" ")
    return total


def _percent_encode(value: str) -> str:
    return urllib.parse.quote(value, safe="~")


def build_oauth1_signature(
    method: str,
    url: str,
    params: Mapping[str, str],
    api_secret: str,
    access_token_secret: str,
) -> str:
    """Compute an RFC 5849 HMAC-SHA1 signature over ``params``."""
    encoded = sorted((_percent_encode(k), _percent_encode(v)) for k, v in params.items())
    param_string = "&".join(f"{k}={v}" for k, v in encoded)
    base_string = "&".join((method.upper(), _percent_encode(url), _percent_encode(param_string)))
    signing_key = f"{_percent_encode(api_secret)}&{_percent_encode(access_token_secret)}"
    digest = hmac.new(signing_key.encode(), base_string.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


def build_oauth1_header(
    method: str,
    url: str,
    credentials: OAuthCredentials,
    nonce: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """Build the ``Authorization`` header for a JSON-body request.

    JSON bodies are not form-encoded, so per OAuth 1.0a only the oauth_*
    parameters (and any query parameters, of which this endpoint has none)
    participate in the signature.
    """
    oauth_params = {
        "oauth_consumer_key": credentials.api_key,
        "oauth_nonce": nonce or _secrets.token_hex(16),
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": timestamp or str(int(time.time())),
        "oauth_token": credentials.access_token,
        "oauth_version": "1.0",
    }
    signature = build_oauth1_signature(
        method, url, oauth_params, credentials.api_secret, credentials.access_token_secret
    )
    oauth_params["oauth_signature"] = signature
    header_params = ", ".join(
        f'{_percent_encode(k)}="{_percent_encode(v)}"' for k, v in sorted(oauth_params.items())
    )
    return f"OAuth {header_params}"


def post_tweet(text: str, credentials: OAuthCredentials) -> str:
    """Post ``text`` via POST /2/tweets and return the created tweet id."""
    body = json.dumps({"text": text}).encode("utf-8")
    request = urllib.request.Request(
        POST_TWEET_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": build_oauth1_header("POST", POST_TWEET_URL, credentials),
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"X API returned {error.code}: {detail}") from error
    return payload["data"]["id"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE_PATH, help="Path to the schedule JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print the due tweet without posting.")
    parser.add_argument("--at", type=str, default=None, help="Override 'now' with an ISO datetime (assumed UTC).")
    parser.add_argument(
        "--lateness-hours",
        type=float,
        default=DEFAULT_LATENESS.total_seconds() / 3600,
        help="How long after its slot a tweet is still considered due.",
    )
    parser.add_argument("--list", action="store_true", help="Print the expanded schedule and exit.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    tweets = load_schedule(args.schedule)

    if args.list:
        for tweet in tweets:
            print(f"{tweet.scheduled_at:%Y-%m-%d %H:%M} UTC  day {tweet.day:>2} {tweet.slot:<8} {tweet.text[:60]}…")
        return 0

    if args.at:
        now = datetime.fromisoformat(args.at)
        now = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
    else:
        now = datetime.now(timezone.utc)

    due = resolve_due_tweet(tweets, now, timedelta(hours=args.lateness_hours))
    if due is None:
        print(f"No tweet due at {now:%Y-%m-%d %H:%M} UTC.")
        return 2

    length = weighted_tweet_length(due.text)
    print(f"Due: day {due.day} / {due.slot} (scheduled {due.scheduled_at:%Y-%m-%d %H:%M} UTC, ~{length} chars)")
    print(due.text)

    if length > MAX_WEIGHTED_LENGTH:
        print(f"ERROR: tweet exceeds the {MAX_WEIGHTED_LENGTH}-character limit ({length}). Not posting.")
        return 1

    if args.dry_run:
        print("Dry run — not posting.")
        return 0

    try:
        credentials = OAuthCredentials.from_env()
    except KeyError as error:
        print(f"ERROR: {error}")
        return 1

    try:
        tweet_id = post_tweet(due.text, credentials)
    except RuntimeError as error:
        print(f"ERROR: {error}")
        return 1

    print(f"Posted tweet id {tweet_id}: https://x.com/i/status/{tweet_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
