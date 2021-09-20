import PIL
from search_engine.search_by_features import SearchEngineByFeature
import streamlit as st
import os
import numpy as np
from streamlit.proto.Image_pb2 import Image
from streamlit.type_util import Key
from face_swapping.face_swapping_cleaned import *
from search_engine.search_by_color import * 
import io
import base64
st.set_page_config(layout="wide")

color_search_engine = None
feature_search_engine = None

def local_css(file_name):
    with open(file_name) as f:
        css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


st.image('./banner1.jpg', use_column_width  = True)
local_css('style.css')
menu = ['Image Based', 'Search Engine']
st.sidebar.header('Mode Selection')
choice = st.sidebar.selectbox('How would you like to be turn ?', menu)

if choice == 'Image Based':

    st.markdown("<h1 style='text-align: center; color: white;'>Face Swapping</h1>", unsafe_allow_html=True)
   
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

    methods = ['Approach 1', 'Approach 2', 'Approach 3']
    approach = cols[1].selectbox('Select an approach', methods)
    cols[2].subheader("")
    if cols[2].button('Swap face'):

        if Image1 is not None and Image2 is not None:
            if approach == 'Approach 1':
                swapped_face = swap_face_approach1(input1, input2)

            if approach == 'Approach 2':
                swapped_face = swap_face_approach2(input1, input2)

            if approach == 'Approach 3':
                swapped_face = swap_face_approach3(input1, input2)

            _, _, swap_col, _, _ = st.columns([1,1,2,1,1])

            swap_col.subheader("Swapped result")
            swap_col.image(swapped_face, use_column_width=True, caption="Swapped Face", channels='BGR')
        else:
            st.warning("Please choose source and destination images")

if choice == 'Search Engine':
    if color_search_engine is None or feature_search_engine is None:
        color_search_engine = HistogramSearch() 
        feature_search_engine = SearchEngineByFeature()
    
    st.markdown("<h1 style='text-align: center; color: white;'>Search Engine</h1>", unsafe_allow_html=True)
    _, col1, _ = st.columns([1, 2, 1])
    
    col1.subheader("Search Image")
    Image1 = col1.file_uploader('Upload image here',type=['jpg','jpeg','png'], key='img1')
    if Image1 is not None:
        Image1 = Image1.read()
        query_image = cv2.imdecode(np.fromstring(Image1, np.uint8), cv2.COLOR_BGR2GRAY)
        col1.image(Image1)
        cols = st.columns([1,1,2,1,1,1])
        search_methods = ['Color', 'Features']
        search_approach = cols[2].selectbox('Select a method', search_methods)
        cols[3].subheader("")
        if cols[3].button('Search'):

            if Image1 is not None :
                if search_approach == 'Color':
                    results = color_search_engine.search(query_image)

                if search_approach == 'Features':
                    results = feature_search_engine.search(get_PIL_from_byte(Image1))

                st.markdown("<h3 style='text-align: left; color: white;'>Search results</h3>", unsafe_allow_html=True)
                r_id = 0  
                tmp_html = ""
                
                for i in range(4):
                    if r_id >= len(results):
                        break
                    images_html = ""
                    for j in range(5):
                        if r_id >= len(results):
                            break
                        images_html = images_html + f"""
                            <img alt="#{r_id}" src="data:image/jpg;base64,{base64.b64encode(open(results[r_id], "rb").read()).decode()}">"""
                        r_id += 1
                    tmp_html = tmp_html + f"""
                    <div class='column'>{images_html}
                    </div>"""
                st.markdown(f"""
                    <div class='row'>{tmp_html}
                    </div> """, unsafe_allow_html=True)
               
            else:
                st.warning("Please choose an image to search!")
