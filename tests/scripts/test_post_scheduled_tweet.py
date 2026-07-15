"""Tests for ``scripts/post_scheduled_tweet.py`` and the tweet schedule data.

Covers schedule loading/validation, due-slot resolution, weighted length
estimation, the OAuth 1.0a signature (against the documented X/Twitter test
vector), and the posting flow with a stubbed HTTP layer.
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pytest  # noqa: E402

from scripts.post_scheduled_tweet import (  # noqa: E402
    DEFAULT_SCHEDULE_PATH,
    MAX_WEIGHTED_LENGTH,
    OAuthCredentials,
    build_oauth1_signature,
    load_schedule,
    main,
    resolve_due_tweet,
    weighted_tweet_length,
)

pytestmark = pytest.mark.unit

UTC = timezone.utc


class TestScheduleData:
    """The committed schedule JSON is valid and complete."""

    def test_campaign_has_63_tweets_over_21_days(self):
        tweets = load_schedule(DEFAULT_SCHEDULE_PATH)
        assert len(tweets) == 63
        assert {t.day for t in tweets} == set(range(1, 22))
        for day in range(1, 22):
            slots = {t.slot for t in tweets if t.day == day}
            assert slots == {"morning", "midday", "evening"}, f"day {day} missing slots"

    def test_schedule_is_sorted_and_unique(self):
        tweets = load_schedule(DEFAULT_SCHEDULE_PATH)
        times = [t.scheduled_at for t in tweets]
        assert times == sorted(times)
        assert len(set(times)) == len(times)

    def test_all_tweets_within_character_limit(self):
        for tweet in load_schedule(DEFAULT_SCHEDULE_PATH):
            length = weighted_tweet_length(tweet.text)
            assert length <= MAX_WEIGHTED_LENGTH, (
                f"day {tweet.day}/{tweet.slot} is {length} weighted chars: {tweet.text!r}"
            )

    def test_all_tweets_nonempty(self):
        for tweet in load_schedule(DEFAULT_SCHEDULE_PATH):
            assert tweet.text.strip()


class TestResolveDueTweet:
    def _tweets(self):
        return load_schedule(DEFAULT_SCHEDULE_PATH)

    def test_exact_slot_time_is_due(self):
        tweets = self._tweets()
        first = tweets[0]
        assert resolve_due_tweet(tweets, first.scheduled_at) == first

    def test_within_lateness_window_is_due(self):
        tweets = self._tweets()
        first = tweets[0]
        due = resolve_due_tweet(tweets, first.scheduled_at + timedelta(hours=1, minutes=59))
        assert due == first

    def test_after_window_is_not_due(self):
        tweets = self._tweets()
        first = tweets[0]
        due = resolve_due_tweet(tweets, first.scheduled_at + timedelta(hours=2))
        assert due != first

    def test_before_campaign_nothing_due(self):
        tweets = self._tweets()
        assert resolve_due_tweet(tweets, tweets[0].scheduled_at - timedelta(minutes=1)) is None

    def test_after_campaign_nothing_due(self):
        tweets = self._tweets()
        assert resolve_due_tweet(tweets, tweets[-1].scheduled_at + timedelta(hours=3)) is None

    def test_each_slot_time_resolves_to_its_own_tweet(self):
        tweets = self._tweets()
        for tweet in tweets:
            assert resolve_due_tweet(tweets, tweet.scheduled_at) == tweet


class TestWeightedTweetLength:
    def test_plain_ascii(self):
        assert weighted_tweet_length("hello world") == 11

    def test_url_counts_as_23(self):
        assert weighted_tweet_length("see https://github.com/Dooders/AgentFarm/very/long/path") == 4 + 23

    def test_emoji_counts_as_two(self):
        assert weighted_tweet_length("hi 🌱") == 3 + 2


class TestOAuth1Signature:
    def test_documented_twitter_test_vector(self):
        """Signature matches the worked example from the X developer docs."""
        params = {
            "status": "Hello Ladies + Gentlemen, a signed OAuth request!",
            "include_entities": "true",
            "oauth_consumer_key": "xvz1evFS4wEEPTGEFPHBog",
            "oauth_nonce": "kYjzVBB8Y0ZFabxSWbWovY3uYSQ2pTgmZeNu2VS4cg",
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": "1318622958",
            "oauth_token": "370773112-GmHxMAgYyLbNEtIKZeRNFsMKPR9EyMZeS9weJAEb",
            "oauth_version": "1.0",
        }
        signature = build_oauth1_signature(
            "POST",
            "https://api.twitter.com/1.1/statuses/update.json",
            params,
            api_secret="kAcSOqF21Fu85e7zjz7ZN2U4ZRhfV3WpwPAoE3Z7kBw",
            access_token_secret="LswwdoUaIvS8ltyTt5jkRh4J50vUPVVHtR2YPi5kE",
        )
        assert signature == "hCtSmYh+iHYCEqBWrE7C7hYmtUk="


class TestCredentials:
    def test_from_env_missing_vars_raises(self):
        with pytest.raises(KeyError, match="X_API_KEY"):
            OAuthCredentials.from_env(env={})

    def test_from_env_reads_all_four(self):
        env = {
            "X_API_KEY": "k",
            "X_API_SECRET": "s",
            "X_ACCESS_TOKEN": "t",
            "X_ACCESS_TOKEN_SECRET": "ts",
        }
        creds = OAuthCredentials.from_env(env=env)
        assert (creds.api_key, creds.api_secret, creds.access_token, creds.access_token_secret) == (
            "k",
            "s",
            "t",
            "ts",
        )


class TestMainCli:
    def _run(self, argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(argv)
        return code, out.getvalue()

    def test_dry_run_at_first_slot(self):
        code, output = self._run(["--dry-run", "--at", "2026-07-20T09:00:00"])
        assert code == 0
        assert "Dry run" in output
        assert "day 1 / morning" in output

    def test_no_tweet_due_returns_2(self):
        code, output = self._run(["--dry-run", "--at", "2026-07-19T00:00:00"])
        assert code == 2
        assert "No tweet due" in output

    def test_list_prints_all_slots(self):
        code, output = self._run(["--list"])
        assert code == 0
        assert output.count("morning") == 21

    def test_posting_uses_stubbed_http(self, monkeypatch, tmp_path):
        for name, value in zip(OAuthCredentials.ENV_VARS, ("k", "s", "t", "ts")):
            monkeypatch.setenv(name, value)

        captured = {}

        class _FakeResponse:
            def read(self):
                return json.dumps({"data": {"id": "12345"}}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode())
            captured["auth"] = request.get_header("Authorization")
            return _FakeResponse()

        monkeypatch.setattr("scripts.post_scheduled_tweet.urllib.request.urlopen", fake_urlopen)

        code, output = self._run(["--at", "2026-07-20T13:30:00"])
        assert code == 0
        assert "Posted tweet id 12345" in output
        assert captured["url"] == "https://api.x.com/2/tweets"
        assert captured["auth"].startswith("OAuth ")
        assert "oauth_signature=" in captured["auth"]
        assert "farm" in captured["body"]["text"].lower()

    def test_now_defaults_to_current_time(self, monkeypatch):
        fixed = datetime(2026, 7, 20, 18, 5, tzinfo=UTC)

        class _FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed

        monkeypatch.setattr("scripts.post_scheduled_tweet.datetime", _FakeDatetime)
        code, output = self._run(["--dry-run"])
        assert code == 0
        assert "day 1 / evening" in output
