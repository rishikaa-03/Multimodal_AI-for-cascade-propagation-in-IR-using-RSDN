"""
Unit tests for auth.py -- the parts that are pure logic, independent of
Streamlit's session/runtime context: password verification and the
role -> permission mapping.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from auth import _verify_password, PERMISSIONS, ROLE_LABELS


def make_hash(password: str, iterations: int = 260_000, salt: bytes = b"fixed-test-salt-16b") -> str:
    """Builds a real PBKDF2-SHA256 hash in the exact 'algorithm$iterations$salt$digest'
    format auth.py expects, so tests exercise the real verification path."""
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations).hex()
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest}"


# ---------------------------------------------------------------------
# _verify_password()
# ---------------------------------------------------------------------

def test_correct_password_verifies():
    stored = make_hash("CorrectHorseBatteryStaple")
    assert _verify_password("CorrectHorseBatteryStaple", stored) is True


def test_wrong_password_fails():
    stored = make_hash("CorrectHorseBatteryStaple")
    assert _verify_password("WrongPassword", stored) is False


def test_empty_password_fails():
    stored = make_hash("CorrectHorseBatteryStaple")
    assert _verify_password("", stored) is False


def test_case_sensitive():
    stored = make_hash("CorrectHorseBatteryStaple")
    assert _verify_password("correcthorsebatterystaple", stored) is False


def test_wrong_algorithm_prefix_rejected():
    """A hash claiming a different algorithm must never verify, even if the
    rest of the string is well-formed -- guards against a downgrade attack
    if the stored format ever changes."""
    digest = hashlib.md5(b"CorrectHorseBatteryStaple").hexdigest()
    stored = f"md5$1$deadbeef${digest}"
    assert _verify_password("CorrectHorseBatteryStaple", stored) is False


def test_malformed_hash_missing_fields_does_not_raise():
    """A corrupted secrets.toml entry must fail closed (return False), not
    crash the login page with an unhandled exception."""
    assert _verify_password("anything", "not-a-valid-hash") is False


def test_malformed_hash_bad_salt_hex_does_not_raise():
    assert _verify_password("anything", "pbkdf2_sha256$1000$not-hex$abcd") is False


def test_malformed_hash_non_integer_iterations_does_not_raise():
    assert _verify_password("anything", "pbkdf2_sha256$notanumber$aabbcc$abcd") is False


def test_different_salts_produce_unverifiable_cross_match():
    """Sanity check that salting is actually happening: the same password
    hashed with two different salts must not cross-verify against the
    wrong stored hash by coincidence."""
    stored_a = make_hash("SamePassword123", salt=b"salt-one-16bytes")
    stored_b = make_hash("SamePassword123", salt=b"salt-two-16bytes")
    assert stored_a != stored_b
    assert _verify_password("SamePassword123", stored_a) is True
    assert _verify_password("SamePassword123", stored_b) is True


# ---------------------------------------------------------------------
# PERMISSIONS / ROLE_LABELS -- structural consistency
# ---------------------------------------------------------------------

def test_every_role_label_has_a_permission_entry():
    assert set(ROLE_LABELS.keys()) == set(PERMISSIONS.keys())


def test_viewer_cannot_run_simulation():
    assert "run_simulation" not in PERMISSIONS["viewer"]
    assert "view_map" in PERMISSIONS["viewer"]


def test_admin_and_controller_have_full_permissions():
    assert PERMISSIONS["admin"] == {"view_map", "run_simulation"}
    assert PERMISSIONS["controller"] == {"view_map", "run_simulation"}


def test_unknown_role_has_no_permissions():
    assert PERMISSIONS.get("nonexistent_role", set()) == set()
