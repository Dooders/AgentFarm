"""Tests for farm/utils/short_id.py (ShortUUID utility)."""
import uuid
import unittest

from farm.utils.short_id import (
    ShortUUID,
    int_to_string,
    string_to_int,
    generate_simulation_id,
    seed,
)


class TestIntToString(unittest.TestCase):
    def test_zero(self):
        alphabet = list("0123456789")
        # 0 with no padding returns empty string reversed = ""
        result = int_to_string(0, alphabet)
        self.assertEqual(result, "")

    def test_simple_number(self):
        # Base-10 alphabet, number 10 → "10"
        alphabet = list("0123456789")
        self.assertEqual(int_to_string(10, alphabet), "10")

    def test_padding_extends_output(self):
        alphabet = list("0123456789")
        result = int_to_string(5, alphabet, padding=4)
        self.assertEqual(len(result), 4)
        self.assertTrue(result.endswith("5"))

    def test_no_padding(self):
        alphabet = list("AB")
        # In base-2 with alphabet ["A","B"]: 3 = "BB"
        self.assertEqual(int_to_string(3, alphabet), "BB")

    def test_large_number(self):
        alphabet = list("0123456789abcdef")
        result = int_to_string(255, alphabet)
        self.assertEqual(result, "ff")


class TestStringToInt(unittest.TestCase):
    def test_empty_string(self):
        alphabet = list("0123456789")
        self.assertEqual(string_to_int("", alphabet), 0)

    def test_single_char(self):
        alphabet = list("0123456789")
        self.assertEqual(string_to_int("5", alphabet), 5)

    def test_multichar(self):
        alphabet = list("0123456789")
        self.assertEqual(string_to_int("10", alphabet), 10)

    def test_binary_alphabet(self):
        alphabet = list("AB")
        # "BB" in base-2 → 3
        self.assertEqual(string_to_int("BB", alphabet), 3)

    def test_round_trip(self):
        alphabet = list("0123456789abcdef")
        for n in [0, 1, 15, 255, 65535]:
            encoded = int_to_string(n, alphabet)
            decoded = string_to_int(encoded, alphabet)
            if n == 0:
                self.assertEqual(decoded, 0)
            else:
                self.assertEqual(decoded, n)


class TestShortUUID(unittest.TestCase):
    def setUp(self):
        self.su = ShortUUID()

    def test_default_alphabet_has_more_than_one_char(self):
        alpha = self.su.get_alphabet()
        self.assertGreater(len(alpha), 1)

    def test_encode_returns_string(self):
        u = uuid.uuid4()
        encoded = self.su.encode(u)
        self.assertIsInstance(encoded, str)
        self.assertGreater(len(encoded), 0)

    def test_encode_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            self.su.encode("not-a-uuid")

    def test_decode_returns_uuid(self):
        u = uuid.uuid4()
        encoded = self.su.encode(u)
        decoded = self.su.decode(encoded)
        self.assertIsInstance(decoded, uuid.UUID)
        self.assertEqual(decoded, u)

    def test_decode_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            self.su.decode(123)

    def test_decode_legacy_mode(self):
        # Legacy mode reverses the string before decoding; test that it processes the reversal
        u = uuid.uuid4()
        encoded = self.su.encode(u)
        # Encode the reversed string so that legacy decode gives back the original UUID
        reversed_encoded = encoded[::-1]
        decoded = self.su.decode(reversed_encoded, legacy=True)
        self.assertIsInstance(decoded, uuid.UUID)
        self.assertEqual(decoded, u)

    def test_uuid_no_name(self):
        result = self.su.uuid()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_uuid_with_name_http(self):
        result = self.su.uuid(name="http://example.com")
        self.assertIsInstance(result, str)
        # Deterministic
        self.assertEqual(result, self.su.uuid(name="http://example.com"))

    def test_uuid_with_name_https(self):
        result = self.su.uuid(name="https://example.com")
        self.assertIsInstance(result, str)
        self.assertEqual(result, self.su.uuid(name="https://example.com"))

    def test_uuid_with_plain_name(self):
        result = self.su.uuid(name="myname")
        self.assertIsInstance(result, str)
        self.assertEqual(result, self.su.uuid(name="myname"))

    def test_uuid_custom_pad_length(self):
        result = self.su.uuid(pad_length=10)
        self.assertIsInstance(result, str)
        self.assertGreaterEqual(len(result), 10)

    def test_random_returns_string(self):
        result = self.su.random()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_random_custom_length(self):
        result = self.su.random(length=8)
        self.assertEqual(len(result), 8)

    def test_get_alphabet(self):
        alpha = self.su.get_alphabet()
        self.assertIsInstance(alpha, str)
        self.assertGreater(len(alpha), 1)

    def test_set_alphabet_too_short_raises(self):
        with self.assertRaises(ValueError):
            self.su.set_alphabet("A")

    def test_set_alphabet_with_duplicates(self):
        # Duplicates are removed
        self.su.set_alphabet("AABBCC")
        alpha = self.su.get_alphabet()
        self.assertEqual(len(alpha), 3)

    def test_encoded_length(self):
        length = self.su.encoded_length(16)
        self.assertIsInstance(length, int)
        self.assertGreater(length, 0)

    def test_id_returns_string(self):
        result = self.su.id()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_uniqueness(self):
        ids = {self.su.uuid() for _ in range(20)}
        self.assertEqual(len(ids), 20)

    def test_length_property(self):
        length = self.su._length
        self.assertIsInstance(length, int)
        self.assertGreater(length, 0)


class TestSeedModule(unittest.TestCase):
    def test_seed_is_short_uuid(self):
        self.assertIsInstance(seed, ShortUUID)


class TestGenerateSimulationId(unittest.TestCase):
    def test_returns_string(self):
        result = generate_simulation_id()
        self.assertIsInstance(result, str)

    def test_default_prefix(self):
        result = generate_simulation_id()
        self.assertTrue(result.startswith("sim_"), f"Expected 'sim_' prefix, got: {result}")

    def test_custom_prefix(self):
        result = generate_simulation_id(prefix="exp")
        self.assertTrue(result.startswith("exp_"), f"Expected 'exp_' prefix, got: {result}")

    def test_uniqueness(self):
        ids = {generate_simulation_id() for _ in range(10)}
        self.assertEqual(len(ids), 10)
