"""
Minimal harness so streamlit.testing.v1.AppTest has an actual script to
run. Not a real page of the dashboard -- just enough to exercise
auth.require_login() end to end (login form -> submit -> authenticated
state -> render_user_bar) in isolation from the rest of app.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import auth

user = auth.require_login()
st.markdown("AUTH_OK")
st.markdown(f"USERNAME:{user['username']}")
st.markdown(f"ROLE:{user['role']}")
st.markdown(f"CAN_SIMULATE:{auth.has_permission('run_simulation')}")
auth.render_user_bar(user)
