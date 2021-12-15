import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image




model=tf.keras.models.load_model("C:\\Users\\lenovo\\PycharmProjects\\potato streamlit\\saved_models\\1")
CLASS_NAMES={ 0:"Early Blight",
              1:"Late Blight",
              2:"Healthy"}
st.title("potato disease classification")
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")




if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is {}".format(CLASS_NAMES [prediction]))



