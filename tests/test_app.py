import streamlit as st

st.title("My First SAiL App")
st.write("Welcome to SAiL!")
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}! This is a test app.")