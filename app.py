#from src.cam_deepface import VideoTransformer
from src.cam_dl import VideoTransformer as VT, sentence, speak
from src.mainpage import mainpage
#from src.deepface_model import deep
import streamlit as st
from PIL import Image
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer
from gtts import gTTS

language = 'en'
def main():

    #Title of the web app
    st.title("Real-time sign language translator")
    
    #Available function in the web app
    activities = ["Homepage","Webcam"]
    choice = st.sidebar.selectbox("Select the required activty to navigate.", activities)

    #Navigating to Homepage
    if choice == "Homepage":
      
      mainpage()

            
    #Navigating to sign language translator with pre=trained deep learning model on live web-cam            
    if choice == "Webcam":

      st.write('Select the respective devices by clicking on SELECT DEVICE option below (Grant access to camera and microphone if needed).')
      st.write('Click on START button to detect hand signs and STOP to end the process.')
      st.write('If the webcam is not working properly, try relaunching the webpage again.')
      webrtc_streamer(key="exam", video_transformer_factory=VT)
      sentence()
      speak()

if __name__ == "__main__":
    main()