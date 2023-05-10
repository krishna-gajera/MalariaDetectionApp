import streamlit as st
from streamlit_jupyter import StreamlitPatcher, tqdm
StreamlitPatcher().jupyter()  # register streamlit with jupyter-compatible wrappers
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('malaria.h5')

# Define the labels
labels = ['Parasitized', 'Uninfected']

# Function to predict the class of the input image
def predict(image):
    # Load the image
    img = Image.open(image).resize((150,150))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    
    # Reshape the image to match the input of the model
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict the class
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    return labels[class_index]

# Define the Streamlit app
def app():
    st.set_page_config(page_title='Malaria Detection', page_icon=':microscope:', layout='wide')
    st.title('Malaria Detection using Image Processing')
    st.write('Upload an image to detect if it is infected with Malaria or not.')
    
    # Create a file uploader
    #uploaded_file = st.file_uploader('Choose an image file', type=['jpg', 'jpeg', 'png'])
    ##Create a form where the image needs to be uploaded :
    uploaded_files = st.file_uploader("Choose an image file", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        st.write(bytes_data)
    
    
    
    '''
    # Make a prediction when an image is uploaded
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write('')
        st.write('Classifying...')
        label = predict(uploaded_file)
        st.wrie(f'This image is {label}')
    '''

if __name__ == '__main__':
    app()
