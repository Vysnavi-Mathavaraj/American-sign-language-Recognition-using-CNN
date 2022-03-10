import cv2
import os
from streamlit_webrtc import VideoTransformerBase
from tensorflow.keras.models import load_model
import numpy as np
from collections import Counter
import streamlit as st
import time
from gtts import gTTS

#loading the pre-trained model file
model = load_model('Final_model.h5')
start_time = time.time()
speech_list=[]
temp=[]
language = 'en'

def sentence():
    st.write("".join(speech_list))

def speak():
    if speech_list != []:
        mytext = ("".join(speech_list))
        myobj = gTTS(text=mytext, lang=language, slow=False)
        myobj.save("text.mp3")
        audio_file = open('text.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)


class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            frame = frame.to_ndarray(format="bgr24")
            labels_dict = {0:'A', 
                 1:'B', 
                 2:'C', 
                 3:'D', 
                 4:'E',
                 5:'F',
                 6:'G',
                 7:'H',
                 8:'I',
                 9:'J',
                 10:'K',
                 11:'L',
                 12:'M',
                 13:'N',
                 14:'O',
                 15:'P',
                 16:'Q',
                 17:'R',
                 18:'S',
                 19:'T',
                 20:'U',
                 21:'V',
                 22:'W',
                 23:'X', 
                 24:'Y', 
                 25:'Z',
                 26:'Delete',
                 27:'Nothing',
                 28:'Space'}  
            '''gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            cv2.rectangle(frame, (24,24), (250,250), (0,255, 0), 2)  
            crop_img=gray[24:250,24:250]
            blur = cv2.GaussianBlur(crop_img,(5,5),2)
            th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            ret, th4 = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            resized=cv2.resize(th4,(64,64))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,64,64,1))'''
            cv2.rectangle(frame, (24,24), (250,250), (0,255, 0), 2)
            crop_img=frame[24:250,24:250]
            resized=cv2.resize(crop_img,(64,64))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,64,64,3))
            result = model.predict(reshaped)
            label=np.argmax(result,axis=1)[0]
            font = cv2.FONT_HERSHEY_SIMPLEX
            if labels_dict[label]!='Delete':
                cv2.putText(frame,labels_dict[label], (275,50), font, 1, (0,0,225), 2, cv2.LINE_4)
                cv2.putText(frame,str(int(time.time() - start_time)%10), (500,50), font, 1, (0,255,225), 2, cv2.LINE_4)
            if int(time.time() - start_time) % 10 == 0:
                if labels_dict[label]=='Space':
                    temp.append(" ")
                elif labels_dict[label]=='Delete':
                    speech_list[:-1]
                elif labels_dict[label]=='Nothing':
                    pass
                else:
                    temp.append(labels_dict[label])
            else:
                try:
                    val = Counter(temp).most_common(1)[0][0]
                    speech_list.append(val)
                    temp.clear()
                except:
                    pass   
            st.write("".join(speech_list))
            return frame



        





'''                
            color_dict=(0,255,0)
            x=0
            y=0
            w=64
            h=64
            img_size=128
            minValue = 70
            count = 0
            string = " "
            prev = " "
            prev_val = 0
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            cv2.rectangle(frame, (24,24), (250,250), (0,255, 0), 2)    
            crop_img=gray[24:250,24:250]
            count = count + 1
            if(count > 0):
                prev_val = count
            cv2.putText(frame, str(prev_val//100), (300, 150),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2) 
            blur = cv2.GaussianBlur(crop_img,(5,5),2)
            th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            th4= cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            resized=cv2.resize(th4,(img_size,img_size))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,img_size,img_size,1))
            result = model.predict(reshaped)
            label=np.argmax(result,axis=1)[0]
            if(label == 0):
                string = string + " "
            else:
                string = string + prev
                
            cv2.putText(frame, prev, (24, 14),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) 
            cv2.putText(frame, string, (275, 50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(200,200,200),2)
            key=cv2.waitKey(1)
            return frame
            


'''


'''

          #Labels for the emotions
          class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
          faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
          
          #Detecting the faces
          

            
            #Drawing rectangle over the face area
            cv2.rectangle(gray), (x,y), (x+w, y+h), (0,255, 0), 2)
            face = gray[ + h, x:x + w]
            face = cv2.resize(face,(48,48))
            face = np.expand_dims(face,axis=0)
            face = face/255.0
            face = face.reshape(face.shape[0],48,48,1)
            
            #Predicting the emotion with the pre-trained model
            preds = model.predict(face)[0]
            label = class_labels[preds.argmax()]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, label, (x,y), font, 1, (0,0,225), 2, cv2.LINE_4)
          
          #returning a frame of the live cam with it's corresponding emotion
          return frame
          '''