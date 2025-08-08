import streamlit as st

def login():
    # Title section with icon
    st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <img src='https://cdn-icons-png.flaticon.com/512/427/427735.png' width='80' alt='Bulb Icon'/>
            <h1 style='color: white; margin-top: 10px;'>MRECW Energy Consumption Dashboard</h1>
        </div>
    """, unsafe_allow_html=True)

    # Centered square login box
    st.markdown("""
        <div style='display: flex; justify-content: center;'>
            <div style='border: 1px solid #444; padding: 30px; width: 320px; border-radius: 15px; background-color: #1e1e1e; box-shadow: 0px 0px 12px rgba(255,255,255,0.1);'>
    """, unsafe_allow_html=True)

    # Form inputs
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Login button
    if st.button("Login"):
        if username and password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Please enter both username and password")

    # Close box
    st.markdown("</div></div>", unsafe_allow_html=True)
