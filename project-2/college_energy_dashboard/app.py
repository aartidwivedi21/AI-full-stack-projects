import streamlit as st
from auth import login
from dashboard import show_dashboard

st.set_page_config(page_title="MRECW Energy Dashboard", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login()
else:
    show_dashboard()
