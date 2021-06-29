import tensorflow as tf
import streamlit as st
import numpy as np
import cv2 
model = tf.keras.models.load_model('/content/drive/MyDrive/Minor project/digit_recognizer.hdf5')
from google.colab.patches import cv2_imshow
from streamlit_drawable_canvas import st_canvas
 
st.title('Digit Recognizer')
 
# Create a canvas component
canvas_result = st_canvas(stroke_width=10,stroke_color='#de2a2a',
    background_color='#000000',
    height=200,width=200,
    drawing_mode='freedraw')
 
# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    img=cv2.resize(canvas_result.image_data.astype('uint8'),(28,28))
    a = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    b=a
    y_pred = model.predict(a.reshape(1,28,28))
    y_pred = np.argmax(y_pred,axis=1)
    if st.button('Predict'):
      st.write(f"output is : {y_pred}")
      st.image(b)
  
  
