"""
System-level test of the login flow using streamlit.testing.v1.AppTest,
which simulates a real running app (form submission, session_state,
reruns) rather than importing functions directly.

Uses a throwaway "tests/_auth_harness_app.py" script and injected
at.secrets["users"] with self-generated test credentials -- your real
.streamlit/secrets.toml and real admin/controller/viewer passwords are
never read or touched by these tests.
"""
import hashlib
import os

import pytest
from streamlit.testing.v1 import AppTest

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "_auth_harness_app.py")


def make_hash(password: str, iterations: int = 1000, salt: bytes = b"test-salt-16byte") -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations).hex()
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest}"


def fresh_app(users: dict) -> AppTest:
    at = AppTest.from_file(HARNESS, default_timeout=15)
    at.secrets["users"] = users
    at.run()
    return at


def test_unauthenticated_shows_login_form_not_dashboard():
    at = fresh_app({"testviewer": {"password_hash": make_hash("testpass123"), "role": "viewer"}})
    assert len(at.text_input) == 2  # username + password fields
    assert not any("AUTH_OK" in m.value for m in at.markdown)


def test_wrong_password_shows_error_and_stays_logged_out():
    at = fresh_app({"testviewer": {"password_hash": make_hash("testpass123"), "role": "viewer"}})
    at.text_input[0].input("testviewer").run()
    at.text_input[1].input("WrongPassword").run()
    at.button[0].click().run()

    assert not any("AUTH_OK" in m.value for m in at.markdown)
    assert len(at.error) == 1
    assert "Invalid username or password" in at.error[0].value


def test_unknown_username_shows_same_generic_error():
    """A different error message for 'unknown user' vs 'wrong password'
    would leak which usernames exist (a user-enumeration issue) -- confirm
    the app deliberately gives one generic message for both."""
    at = fresh_app({"testviewer": {"password_hash": make_hash("testpass123"), "role": "viewer"}})
    at.text_input[0].input("nosuchuser").run()
    at.text_input[1].input("whatever").run()
    at.button[0].click().run()

    assert len(at.error) == 1
    assert "Invalid username or password" in at.error[0].value


def test_correct_password_authenticates_with_correct_role():
    at = fresh_app({"testviewer": {"password_hash": make_hash("testpass123"), "role": "viewer"}})
    at.text_input[0].input("testviewer").run()
    at.text_input[1].input("testpass123").run()
    at.button[0].click().run()

    values = [m.value for m in at.markdown]
    assert "AUTH_OK" in values
    assert "USERNAME:testviewer" in values
    assert "ROLE:viewer" in values


def test_viewer_role_cannot_run_simulation():
    at = fresh_app({"testviewer": {"password_hash": make_hash("testpass123"), "role": "viewer"}})
    at.text_input[0].input("testviewer").run()
    at.text_input[1].input("testpass123").run()
    at.button[0].click().run()

    assert "CAN_SIMULATE:False" in [m.value for m in at.markdown]


def test_controller_role_can_run_simulation():
    at = fresh_app({"testctrl": {"password_hash": make_hash("ctrlpass456"), "role": "controller"}})
    at.text_input[0].input("testctrl").run()
    at.text_input[1].input("ctrlpass456").run()
    at.button[0].click().run()

    assert "CAN_SIMULATE:True" in [m.value for m in at.markdown]


def test_admin_role_can_run_simulation():
    at = fresh_app({"testadmin": {"password_hash": make_hash("adminpass789"), "role": "admin"}})
    at.text_input[0].input("testadmin").run()
    at.text_input[1].input("adminpass789").run()
    at.button[0].click().run()

    assert "CAN_SIMULATE:True" in [m.value for m in at.markdown]


def test_username_lookup_is_case_insensitive():
    at = fresh_app({"testviewer": {"password_hash": make_hash("testpass123"), "role": "viewer"}})
    at.text_input[0].input("TestViewer").run()  # mixed case, on purpose
    at.text_input[1].input("testpass123").run()
    at.button[0].click().run()

    assert "USERNAME:testviewer" in [m.value for m in at.markdown]


def test_account_with_invalid_role_is_rejected():
    """A secrets.toml entry with a typo'd/unknown role must not silently
    grant access -- guards against a config mistake becoming a privilege
    escalation."""
    at = fresh_app({"broken": {"password_hash": make_hash("whatever"), "role": "superadmin"}})
    at.text_input[0].input("broken").run()
    at.text_input[1].input("whatever").run()
    at.button[0].click().run()

    assert not any("AUTH_OK" in m.value for m in at.markdown)
    assert len(at.error) == 1
    assert "invalid role" in at.error[0].value.lower()
