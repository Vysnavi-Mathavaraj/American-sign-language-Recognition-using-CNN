import streamlit as st

#Contents displayed on the Homepage
def mainpage():
    st.write(" American Sign Language Translator app detects hand signs and converts it into text. It can detect 26 Alphabets along with space.")
    st.write('This app uses deep learning model to predict the hand signs in live webcam.')
    st.write('You can select webcam from the dropdown to navigate to the page.')