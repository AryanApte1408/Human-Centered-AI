
# streamlit_app.py
import streamlit as st

st.set_page_config(page_title="HW manager", layout="wide")
st.title("HW manager")

st.sidebar.title("Navigate")
page = st.sidebar.radio("Select Homework:", ["HW1", "HW2"], index=0)

if page == "HW1":
    from HWs.HW1 import app as hw1_app
    hw1_app()
else:
    from HWs.HW2 import app as hw2_app
    hw2_app()


