# streamlit_app.py
import streamlit as st

st.set_page_config(page_title="HW manager", layout="wide")
st.title("HW manager")

st.sidebar.title("Navigate")
page = st.sidebar.radio("Select Homework:", ["HW1", "HW2", "HW3"], index=0)

if page == "HW1":
    from HWs.HW1 import app as hw1_app
    hw1_app()

elif page == "HW2":
    from HWs.HW2 import app as hw2_app
    hw2_app()

elif page == "HW3":
    from HWs.HW3 import app as hw3_app
    hw3_app()
