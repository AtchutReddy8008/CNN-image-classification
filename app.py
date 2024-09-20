import tensorflow as tf
import streamlit as st
from PIL import Image
import os
import numpy as np

model=tf.keras.models.load_model("model.h5")

st.title("Classification using CNN")

def pred_process(upload_img):
    img=Image.open(upload_img).convert("RGB")
    
    img=img.resize((128,128))
    img_array=np.array(img)
    img_array=np.expand_dims(img_array,axis=0)/255.0
    
    prediction_prob=model.predict(img_array)
    prediction_class=np.argmax(prediction_prob,axis=1)
    return prediction_prob,prediction_class





upload_img=st.file_uploader("upload your IMG",type=["jpg","png","jpeg"])
if upload_img is not None:
    st.image(upload_img,caption="Your uploaded img is",use_column_width=True)
    pred_prob,pred_class=pred_process(upload_img)
    classes={0:"allu arjun",1:"ntr",2:"prabhas",3:"ram chran"}
    pred_classs=classes[pred_class[0]]
    
    st.write(f"prediction class :  {pred_classs}")
    st.write(f"prediction probability :  {pred_prob}")