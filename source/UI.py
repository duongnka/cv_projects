import streamlit as st
import os
import numpy as np
from streamlit.proto.Image_pb2 import Image
from streamlit.type_util import Key
from face_swapping_cleaned import *

st.set_page_config(layout="wide")

st.image('./image_data/banner1.jpg', use_column_width  = True)
st.markdown("<h1 style='text-align: center; color: white;'>Future Vision !</h1>", unsafe_allow_html=True)

menu = ['Image Based']
st.sidebar.header('Mode Selection')
choice = st.sidebar.selectbox('How would you like to be turn ?', menu)

if choice == 'Image Based':
    
    st.sidebar.header('Configuration')
    _, col1, col2, _ = st.columns([1, 2, 2, 1])
    
    
    col1.subheader("Source image")
    Image1 = col1.file_uploader('Upload image here',type=['jpg','jpeg','png'], key='img1')

    col2.subheader("Destination image")
    Image2 = col2.file_uploader('Upload image here',type=['jpg','jpeg','png'], key='img2')

    if Image1 is not None:
        Image1 = Image1.read()
        input1 = cv2.imdecode(np.fromstring(Image1, np.uint8), cv2.COLOR_BGR2GRAY)
        col1.image(Image1, use_column_width=True)

    if Image2 is not None:
        Image2 = Image2.read()
        input2 = cv2.imdecode(np.fromstring(Image2, np.uint8), cv2.COLOR_BGR2GRAY)
        col2.image(Image2, use_column_width=True)

    cols = st.columns([1,2,1,1,1])

    methods = ['Approach 1', 'Approach 2']
    approach = cols[1].selectbox('Select an approach', methods)
    cols[2].subheader("")
    if cols[2].button('Swap face'):

        if Image1 is not None and Image2 is not None:
            if approach == 'Approach 1':
                swapped_face = swap_face(input1, input2, True)
            else:
                swapped_face = swap_face(input1, input2, False)

            _, _, swap_col, _, _ = st.columns([1,1,2,1,1])

            swap_col.subheader("Swapped result")
            swap_col.image(swapped_face, use_column_width=True, caption="Swapped Face", channels='BGR')
        else:
            st.warning("Please choose source and destination images")