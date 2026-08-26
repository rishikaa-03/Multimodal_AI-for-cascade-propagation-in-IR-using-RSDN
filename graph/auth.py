"""
Database-free authentication and role-based authorization for Railway Cascade AI.

Roles:
    admin      -> full access
    controller -> network map + live simulation
    viewer     -> network map only

Credentials are stored as PBKDF2-SHA256 hashes in .streamlit/secrets.toml.
"""

import hashlib
import hmac
from datetime import datetime

import streamlit as st


ROLE_LABELS = {
    "admin": "System Administrator",
    "controller": "Railway Controller",
    "viewer": "Read-Only Viewer",
}

PERMISSIONS = {
    "admin": {"view_map", "run_simulation"},
    "controller": {"view_map", "run_simulation"},
    "viewer": {"view_map"},
}


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a PBKDF2-SHA256 password hash."""
    try:
        algorithm, iterations, salt_hex, digest_hex = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False

        iterations = int(iterations)
        salt = bytes.fromhex(salt_hex)

        calculated = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        ).hex()

        return hmac.compare_digest(calculated, digest_hex)
    except (ValueError, TypeError):
        return False


def _users():
    """Read user records from Streamlit secrets."""
    try:
        return st.secrets["users"]
    except Exception:
        return {}


def current_user():
    """Return the authenticated user record or None."""
    username = st.session_state.get("username")
    role = st.session_state.get("role")

    if not username or not role:
        return None

    return {
        "username": username,
        "role": role,
        "display_name": ROLE_LABELS.get(role, role.title()),
        "role_label": ROLE_LABELS.get(role, role.title()),
    }


def has_permission(permission: str) -> bool:
    """Check whether the current user has a permission."""
    user = current_user()
    return bool(user and permission in PERMISSIONS.get(user["role"], set()))


def logout():
    """Clear authentication state."""
    for key in ("authenticated", "username", "role"):
        st.session_state.pop(key, None)



def _login_page():
    """Render a professional, wide Railway Cascade AI login screen."""

    st.markdown(
        """
        <style>
        /* ===== LOGIN PAGE ONLY ===== */
        /* Keep Streamlit's native toolbar/settings menu visible. */
        header[data-testid="stHeader"] { visibility: visible !important; }
        /* Match the dashboard's wide laptop layout. */
        .block-container,
        [data-testid="stMainBlockContainer"] {
            max-width: 1800px !important;
            width: 100% !important;
            box-sizing: border-box !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-top: 1.15rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding-bottom: 3rem !important;
        }

        /* ---------- Exact dashboard-style Railway system header ---------- */
        .railway-header {
            position: relative;
            overflow: hidden;
            width: 100%;
            min-height: 94px;
            box-sizing: border-box;
            margin: 0 0 28px 0;
            padding: 18px 26px;
            border-radius: 12px;
            border-bottom: 3px solid #ef3030;
            background: linear-gradient(110deg, #071b31 0%, #123b61 55%, #8f2028 100%);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.14);
            display: flex;
            align-items: center;
            gap: 18px;
            text-align: left !important;
        }

        .railway-header::before {
            content: "";
            position: absolute;
            right: 12px;
            top: -76px;
            width: 175px;
            height: 175px;
            border-radius: 50%;
            border: 30px solid rgba(255,255,255,0.055);
            pointer-events: none;
        }

        .railway-header::after {
            content: "";
            position: absolute;
            right: -18px;
            bottom: -92px;
            width: 140px;
            height: 140px;
            border-radius: 50%;
            border: 24px solid rgba(255,255,255,0.045);
            pointer-events: none;
        }

        .system-icon {
            position: relative;
            z-index: 2;
            flex: 0 0 48px;
            width: 48px;
            height: 48px;
            border: 1px solid rgba(255,255,255,0.55);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(255,255,255,0.035);
            color: #ffffff;
            font-size: 23px;
            line-height: 1;
        }

        .system-copy {
            position: relative;
            z-index: 2;
            min-width: 0;
            flex: 1 1 auto;
        }

        .railway-title {
            display: block !important;
            margin: 0 !important;
            padding: 0 !important;
            color: #ffffff !important;
            font-size: 25px !important;
            font-weight: 800 !important;
            line-height: 1.15 !important;
            text-align: left !important;
            white-space: nowrap !important;
        }

        .railway-subtitle {
            display: block !important;
            margin: 6px 0 0 0 !important;
            padding: 0 !important;
            color: #d7e2ef !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            line-height: 1.35 !important;
            text-align: left !important;
            white-space: nowrap !important;
        }

        .system-status {
            position: relative;
            z-index: 2;
            flex: 0 0 auto;
            align-self: flex-start;
            margin-left: auto;
            text-align: right;
        }

        .status-pill {
            display: inline-block !important;
            margin: 0 !important;
            padding: 7px 13px !important;
            border-radius: 4px !important;
            border: 1px solid rgba(133, 255, 159, 0.55) !important;
            background: rgba(55, 120, 62, 0.18) !important;
            color: #91e89e !important;
            font-size: 11px !important;
            font-weight: 750 !important;
            letter-spacing: .6px !important;
            line-height: 1.2 !important;
            white-space: nowrap !important;
        }

        .system-time {
            display: block;
            margin-top: 8px;
            color: #e3e7ed !important;
            font-size: 10px;
            font-weight: 550;
            white-space: nowrap;
        }

        .login-panel {
            max-width: 650px;
            margin: 24px auto 0 auto;
        }

        /* The Streamlit form itself becomes the login card. */
        div[data-testid="stForm"] {
            border: 1px solid var(--st-border-color, #d9dee7) !important;
            border-radius: 14px !important;
            padding: 28px 30px 25px 30px !important;
            background: var(--st-secondary-background-color) !important;
            box-shadow: 0 10px 28px rgba(0,0,0,.09) !important;
        }

        .login-heading {
            max-width: 650px;
            margin: 24px auto 0 auto;
            padding: 0 4px;
            color: var(--st-text-color);
            font-size: 21px;
            font-weight: 750;
        }

        .login-description {
            max-width: 650px;
            margin: 7px auto 0 auto;
            padding: 0 4px;
            color: var(--st-text-color);
            font-size: 13px;
        }

        div[data-testid="stTextInput"] label,
        div[data-testid="stTextInput"] label p {
            color: var(--st-text-color) !important;
            opacity: 1 !important;
            font-weight: 650 !important;
        }

        div[data-testid="stTextInput"] input {
            border-radius: 8px !important;
            min-height: 46px !important;
            background-color: var(--st-secondary-background-color) !important;
            color: var(--st-text-color) !important;
            -webkit-text-fill-color: var(--st-text-color) !important;
            border-color: var(--st-border-color) !important;
        }

        div[data-testid="stTextInput"] input::placeholder {
            color: var(--st-text-color) !important;
            opacity: 0.65 !important;
        }

        div[data-testid="stFormSubmitButton"] button {
            min-height: 47px !important;
            border-radius: 8px !important;
            border: 0 !important;
            background: linear-gradient(135deg, #c62828, #9e1b1b) !important;
            color: #ffffff !important;
            font-weight: 750 !important;
            letter-spacing: .2px;
            box-shadow: 0 4px 10px rgba(183,28,28,.22);
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            background: linear-gradient(135deg, #d63131, #aa1d1d) !important;
            color: #ffffff !important;
            transform: translateY(-1px);
        }

        .login-security {
            display: flex;
            justify-content: center;
            gap: 28px;
            flex-wrap: wrap;
            max-width: 800px;
            margin: 20px auto 0 auto;
            color: var(--st-text-color);
            font-size: 11px;
            font-weight: 650;
            letter-spacing: .4px;
            text-transform: uppercase;
        }

        .login-security span::before {
            content: "●";
            color: #2e9b57;
            margin-right: 6px;
        }

        .login-footer {
            margin-top: 15px;
            text-align: center;
            color: var(--st-text-color);
            font-size: 11px;
        }

        @media (max-width: 900px) {
            .block-container,
            [data-testid="stMainBlockContainer"] {
                max-width: 100% !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }

            .railway-header {
                min-height: auto;
                padding: 16px 18px;
                gap: 12px;
                align-items: flex-start;
            }

            .system-icon {
                flex-basis: 40px;
                width: 40px;
                height: 40px;
                font-size: 19px;
            }

            .railway-title {
                font-size: 20px !important;
                white-space: normal !important;
            }

            .railway-subtitle {
                font-size: 11px !important;
                white-space: normal !important;
            }

            .system-status {
                display: none;
            }

            .login-panel {
                max-width: 100%;
            }

            div[data-testid="stForm"] {
                padding: 22px 20px 20px 20px !important;
            }

            .login-heading,
            .login-description {
                max-width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Wide branded header — matches the final dashboard app.py.
    st.markdown(
        f"""
        <div class="railway-header">
            <div class="system-icon">🚆</div>
            <div class="system-copy">
                <div class="railway-title">Railway Cascade AI System</div>
                <div class="railway-subtitle">
                    Cross-Zone Cascade Frontier Detection &amp; Network Risk Intelligence
                </div>
            </div>
            <div class="system-status">
                <div class="status-pill">System Operational</div>
                <div class="system-time">
                    {datetime.now().astimezone().strftime("%d %b %Y, %H:%M %Z")}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="login-heading">🔐 Secure Operations Portal</div>'
        '<div class="login-description">'
        'Sign in with your authorized railway operations account.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Keep the actual Streamlit form centered and professional.
    _, center, _ = st.columns([1.0, 1.6, 1.0], gap="large")
    with center:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                autocomplete="username",
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                autocomplete="current-password",
            )
            submitted = st.form_submit_button(
                "🔐  SIGN IN",
                type="primary",
                use_container_width=True,
            )

    st.markdown(
        """
        <div class="login-security">
            <span>Secure Access</span>
            <span>AI Monitoring</span>
            <span>Authorized Users</span>
        </div>
        <div class="login-footer">
            Railway Cascade AI • Cross-Zone Cascade Frontier Detection
        </div>
        """,
        unsafe_allow_html=True,
    )

    if submitted:
        username = username.strip().lower()
        user_store = _users()
        record = user_store.get(username)

        if record and _verify_password(password, record["password_hash"]):
            role = record["role"]

            if role not in PERMISSIONS:
                st.error("This account has an invalid role configuration.")
                st.stop()

            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.role = role
            st.rerun()
        else:
            st.error("Invalid username or password.")


def require_login():
    """Require authentication before the protected dashboard loads."""
    if not st.session_state.get("authenticated", False):
        _login_page()
        st.stop()

    user = current_user()

    if user is None:
        logout()
        _login_page()
        st.stop()

    return user


def render_user_bar(user):
    """Show the current user and a logout control."""
    left, right = st.columns([8, 1])

    with left:
        st.caption(
            f"👤 **{user['display_name']}**  •  "
            f"Role: **{user['role_label']}**"
        )

    with right:
        if st.button("Logout", key="logout_button"):
            logout()
            st.rerun()