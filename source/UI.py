import streamlit as st
import os
import numpy as np
from streamlit.proto.Image_pb2 import Image
from streamlit.type_util import Key
from utils_face_processing import *

    

st.set_page_config(layout="wide")

st.image('./image_data/banner1.jpg', use_column_width  = True)
st.markdown("<h1 style='text-align: center; color: white;'>Future Vision !</h1>", unsafe_allow_html=True)

# menu = ['Image Based', 'Video Based']
menu = ['Image Based']
st.sidebar.header('Mode Selection')
choice = st.sidebar.selectbox('How would you like to be turn ?', menu)
# Create the Home page
if choice == 'Image Based':
    
    st.sidebar.header('Configuration')
    # outputsize = st.sidebar.selectbox('Output Size', [384,512,768])
    # Autocrop = st.sidebar.checkbox('Auto Crop Image',value=True) 
    # gamma = st.sidebar.slider('Gamma adjust', min_value=0.1, max_value=3.0,value=1.0,step=0.1) # change the value here to get different result
    _, col1, col2, _ = st.columns([1, 2, 2, 1])
    
    
    col1.subheader("Image 1")
    Image1 = col1.file_uploader('Upload image here',type=['jpg','jpeg','png'], key='img1')

    col2.subheader("Image 2")
    Image2 = col2.file_uploader('Upload image here',type=['jpg','jpeg','png'], key='img2')

    if Image1 is not None:
        Image1 = Image1.read()
        input1 = cv2.imdecode(np.fromstring(Image1, np.uint8), cv2.COLOR_BGR2GRAY)
        col1.image(Image1, use_column_width=True)

    if Image2 is not None:
        Image2 = Image2.read()
        input2 = cv2.imdecode(np.fromstring(Image2, np.uint8), cv2.COLOR_BGR2GRAY)
        col2.image(Image2, use_column_width=True)
    cols = st.columns([1,1,1,1,1,1,1,1,1])
    if cols[4].button('Swap face'):
        if Image1 is not None and Image2 is not None:
            img, img_gray, mask = process_input_img(input1)
            img2, img2_gray, mask2 = process_input_img(input2)

            swapped_face1 = swap_face(img, img_gray, img2, img2_gray)
            swapped_face2 = swap_face(img2, img2_gray, img, img_gray)

            _, swap_col1, swap_col2, _ = st.columns([1,2,2,1])

            swap_col1.subheader("Swapped result")
            swap_col1.image(swapped_face2, use_column_width=True, caption="Face 1 to Face 2", channels='BGR')
            
            swap_col2.subheader("Swapped result")
            swap_col2.image(swapped_face1, use_column_width=True, caption="Face 2 to Face 1", channels='BGR')