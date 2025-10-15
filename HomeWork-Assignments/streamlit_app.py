# streamlit_app.py
import streamlit as st

st.set_page_config(page_title="HW Manager", layout="wide")
st.title("HW Manager")

st.sidebar.title("Navigate")
page = st.sidebar.radio(
    "Select Homework:",
    ["HW1", "HW2", "HW3", "HW4", "HW5", "HW7"],  # ✅ Added HW5
    index=0
)

if page == "HW1":
    from HWs.HW1 import app as hw1_app
    hw1_app()

elif page == "HW2":
    from HWs.HW2 import app as hw2_app
    hw2_app()

elif page == "HW3":
    from HWs.HW3 import app as hw3_app
    hw3_app()

elif page == "HW4":
    from HWs.HW4 import app as hw4_app
    hw4_app()

elif page == "HW5":  
    from HWs.HW5 import app as hw5_app
    hw5_app()

elif page == "HW7":
    from HWs.HW7 import app as hw7_app
    hw7_app()
