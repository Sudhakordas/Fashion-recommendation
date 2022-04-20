import pickle
import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

from numpy.linalg import norm

from tensorflow.keras.models import load_model
from PIL import Image

model = load_model('fashion_recommendation_model.h5')
file_names = pickle.load(open('file_names.pkl', 'rb'))
feature_list = np.array(pickle.load(open('features_list.pkl', 'rb')))

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

    
def feature_extraction(image_path, model):
    img = image.load_img(image_path, target_size = (224,224))
    img_array = image.img_to_array(img)
    expanded_image = np.expand_dims(img_array, axis = 0)
    preprocessed_image = preprocess_input(expanded_image)
    result = model.predict(preprocessed_image).flatten()
    normalised_image =result / norm(result)
    
    return normalised_image

def recommend(features, feature_list):
    Neighbors =NearestNeighbors(5, algorithm = 'brute', metric = 'euclidean')
    Neighbors.fit(feature_list)

    distance, indices = Neighbors.kneighbors([features])
    
    return indices

st.title('Fashion Recommeder System')

#Uploading the file
uploaded_file = st.file_uploader('Choose a file')
if uploaded_file is not None:
    
    if save_uploaded_file(uploaded_file):
        #file has been uploaded
        #display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        #getting the features
        features = feature_extraction(os.path.join('uploads',uploaded_file.name), model)
        #st.text(features)
        #recommendation
        indices = recommend(features, feature_list)
        col1, col2, col3, col4, col5  = st.columns(5)
        
        with col1:
            st.image(file_names[indices[0][0]])
            
        with col2:
            st.image(file_names[indices[0][1]])
        
        with col3:
            st.image(file_names[indices[0][2]])
        
        with col4:
            st.image(file_names[indices[0][3]])
        
        with col5:
            st.image(file_names[indices[0][4]])
        

        

    else:
        st.header('Some error occured')
