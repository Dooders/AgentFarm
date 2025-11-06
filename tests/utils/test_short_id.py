"""Unit tests for `farm.utils.short_id`."""

from __future__ import annotations

import os
import uuid
from unittest import mock

import pytest

from farm.utils import short_id


def test_int_to_string_and_string_to_int_round_trip() -> None:
    alphabet = list("0123456789ABCDEF")
    number = 0xABCD

    encoded = short_id.int_to_string(number, alphabet, padding=6)
    assert encoded == "00ABCD"
    decoded = short_id.string_to_int(encoded, alphabet)
    assert decoded == number


def test_int_to_string_padding_short_numbers() -> None:
    alphabet = list("01")
    encoded = short_id.int_to_string(3, alphabet, padding=6)
    # 3 in binary is 11, expect left-padded to 6 characters using alphabet[0]
    assert encoded == "000011"


def test_shortuuid_encode_decode_round_trip() -> None:
    short = short_id.ShortUUID()
    original = uuid.uuid4()
    encoded = short.encode(original)
    decoded = short.decode(encoded)
    assert decoded == original


def test_shortuuid_encode_requires_uuid() -> None:
    short = short_id.ShortUUID()
    with pytest.raises(ValueError):
        short.encode("not-a-uuid")  # type: ignore[arg-type]


def test_shortuuid_decode_requires_string() -> None:
    short = short_id.ShortUUID()
    with pytest.raises(ValueError):
        short.decode(1234)  # type: ignore[arg-type]


def test_shortuuid_uuid_with_name_is_deterministic() -> None:
    short = short_id.ShortUUID()
    value_one = short.uuid(name="example.com")
    value_two = short.uuid(name="example.com")
    assert value_one == value_two


def test_shortuuid_uuid_uses_url_namespace_for_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    short = short_id.ShortUUID()

    # Force uuid5 to return a known UUID so we can assert the encoded form exactly.
    fake_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")

    with mock.patch.object(uuid, "uuid5", return_value=fake_uuid) as patched_uuid5:
        result = short.uuid(name="https://example.com", pad_length=short._length)

    patched_uuid5.assert_called_once()
    # Ensure the encoded output decodes back into the fake UUID we supplied.
    assert short.decode(result) == fake_uuid


def test_shortuuid_random_respects_length_and_alphabet(monkeypatch: pytest.MonkeyPatch) -> None:
    short = short_id.ShortUUID()
    alphabet = set(short.get_alphabet())

    # Provide deterministic bytes so the test is reproducible.
    monkeypatch.setattr(os, "urandom", lambda n: b"\xff" * n)

    random_value = short.random(length=8)
    assert len(random_value) == 8
    assert set(random_value).issubset(alphabet)


def test_set_alphabet_validates_unique_symbols() -> None:
    short = short_id.ShortUUID()
    with pytest.raises(ValueError):
        short.set_alphabet("a")


def test_set_alphabet_sorts_and_deduplicates() -> None:
    custom = short_id.ShortUUID(alphabet="cbaacb")
    assert custom.get_alphabet() == "abc"


def test_encoded_length_matches_expected() -> None:
    short = short_id.ShortUUID()
    # encoded_length should match the internal length for 16 bytes (standard UUID size)
    assert short.encoded_length(16) == short._length


def test_generate_simulation_id_delegates_to_identity() -> None:
    with mock.patch("farm.utils.identity.Identity") as identity_cls:
        identity_instance = identity_cls.return_value
        identity_instance.simulation_id.return_value = "prefix_custom"

        result = short_id.generate_simulation_id(prefix="prefix")

    identity_cls.assert_called_once_with()
    identity_instance.simulation_id.assert_called_once_with("prefix")
    assert result == "prefix_custom"
