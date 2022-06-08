import streamlit as st
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from PIL import Image
import os
import joblib
import tensorflow as tf
import pickle


#[theme]
#base = 'dark'
#primaryColor = 'green'

st.title ("Skin Cancer Detection")

st.write("This website aims to develop a skin cancer detection based on dermatoscopic images and a patient's metadata using a Deep Learning model.")

with st.expander("The Process of Skin Cancer Detection"):
    st.write("The process of skin cancer detection has significantly improved over the last years. Many different")
    st.write(" techniques have been applied. Particularly, the method of image classification has taken the")
    st.write("accuracy of the diagnosis to a whole new level.")
    st.write("This model classifies the input as one out of seven diagnostic categories of pigmented lesions.")

col1,col2 = st.columns(2)

with col1:
    st.radio("Choose your Gender", ["male", "female"])
    st.markdown('#')

    st.slider("Choose your Age", 0, 100)
    st.markdown('#')

    location = st.selectbox ("Where is your spot localized?", ['', 'Scalp', 'Ear', 'Face', 'Back', 'Trunk', 'Chest',
       'Upper Extremity', 'Abdomen', 'Lower Extremity',
       'Genital', 'Neck', 'Hand', 'Foot', 'Acral'])
    #with st.container('If Other applies:'):
        #st.write('')

with col2:
    with st.expander("Option 1: Upload a Photo"):
        uploaded_file = st.file_uploader("Insert your Photo here")
        if uploaded_file:
            st.image(uploaded_file, width = 228)

    st.markdown('#')
    st.markdown('#')

    with st.expander("Option 2: Take a Photo with your camera"):
        camera_file = st.camera_input ("")
        if camera_file:
            st.image(camera_file, width = 225)

st.markdown('#')

agree = st.checkbox("*I am aware that this model cannot replace the assessment of a doctor.")
if agree:
    st.markdown('#')

    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        pass
    with col2:
        pass
    #with col3:
    #    st.button("Submit")
    with col4:
        pass
    with col5:
        pass


    if st.button('Check my lesion'):
        if uploaded_file is not None:
            X_input = np.asarray(Image.open(uploaded_file).resize((224,224)))
            #X_input = np.asarray(Image.open(uploaded_file))
            #st.markdown(f'### predicted type: {X_input.shape}')
            #X_input_stack = np.stack(X_input)
        else:
            X_input = np.asarray(Image.open(camera_file).resize((224,224)))
        X_input_stack = np.reshape(X_input,(1,224,224,3))
        #st.markdown(f'### predicted type: {X_input_stack.shape}')

        joblib_model = joblib.load('basic_model1_224_augmented_epoch45.joblib')
        # loaded_model = pickle.load(open('basic_with_aug_model', 'rb'))
        cancer_type = joblib_model.predict(X_input_stack)
        probability = cancer_type
            #[np.argmax(cancer_type)]*100
        CLASSES = ["benign keratosis-like lesions", "melanocytic nevi", "dermatofibroma", "melanoma", "vascular lesions", "basal cell carcinoma", "Actinic keratoses and intraepithelial carcinoma / Bowen's disease"]
        endresult = CLASSES[np.argmax(cancer_type)]
        st.markdown(f'### predicted type: {endresult}')
        # st.markdown(f'### predicted type: {endresult}, probability:{probability}')
        if endresult == 'benign keratosis-like lesions':
            st.markdown('These include solar lentigines / seborrheic keratoses and lichen-planus like keratoses')
            image = Image.open('green.jpg')
            col1, col2, col3 = st.columns([1,6,1])
            with col1:
                st.write("")
            with col2:
                st.image(image, width=(200))
            with col3:
                st.write("")
            st.markdown("They appear gradually, usually on the face, neck, chest or back. Seborrheic keratoses are harmless and not contagious. They don't need treatment, but you may decide to have them removed if they become irritated by clothing or you don't like how they look. (Source: mayoclinic.org")
        if endresult == 'vascular lesions':
            st.markdown('These include angiomas, angiokeratomas, pyogenic granulomas and hemorrhage')
            image = Image.open('green.jpg')
            col1, col2, col3 = st.columns([1,6,1])
            with col1:
                st.write("")
            with col2:
                st.image(image, width=(200))
            with col3:
                st.write("")
            st.markdown("Vascular lesions are relatively common abnormalities of the skin and underlying tissues, more commonly known as birthmarks. (Source: ssmhealth.com)")
        if endresult == 'melanoma':
            image = Image.open('red.jpg')
            col1, col2, col3 = st.columns([1,6,1])
            with col1:
                st.write("")
            with col2:
                st.image(image, width=(200))
            with col3:
                st.write("")
            st.markdown("Melanoma is a serious form of skin cancer that begins in cells known as melanocytes. While it is less common than basal cell carcinoma (BCC) and squamous cell carcinoma (SCC), melanoma is more dangerous because of its ability to spread to other organs more rapidly if it is not treated at an early stage. (Source: skincancer.org)")
        if endresult == 'melanocytic nevi':
            image = Image.open('green.jpg')
            col1, col2, col3 = st.columns([1,6,1])
            with col1:
                st.write("")
            with col2:
                st.image(image, width=(200))
            with col3:
                st.write("")
            st.markdown("A melanocytic naevus, or mole, is a common benign skin lesion due to a local proliferation of pigment cells (melanocytes). A brown or black melanocytic naevus contains the pigment melanin, so may also be called a pigmented naevus. It can be present at birth (a congenital melanocytic naevus) or appear later (an acquired naevus). (Source: dermnetnz.org)")
        if endresult == 'dermatofibroma':
            image = Image.open('green.jpg')
            col1, col2, col3 = st.columns([1,6,1])
            with col1:
                st.write("")
            with col2:
                st.image(image, width=(200))
            with col3:
                st.write("")
            st.markdown("Dermatofibromas are common noncancerous (benign) skin growths. They are firm to hard, and they are skin-colored or slightly pigmented. These lesions usually persist for life, and they may heal as depressed scars after several years. (Source: skinsight.com)")
        if endresult == 'basal cell carcinoma':
            image = Image.open('red.jpg')
            col1, col2, col3 = st.columns([1,6,1])
            with col1:
                st.write("")
            with col2:
                st.image(image, width=(200))
            with col3:
                st.write("")
            st.markdown("A basal cell carcinoma is a cancerous skin lesion and the most common type of skin cancer. It most often develops in skin areas which have been exposed to direct sunlight. Although it grows slowly and seldomly spreads to another part of the body, treatment is important. Given time to grow, this skin cancer can grow deep, injuring nerves, blood vessels, and anything else in its path. (Source: aad.org)")
        if endresult == "Actinic keratoses and intraepithelial carcinoma / Bowen's disease":
            image = Image.open('yellow.jpg')
            col1, col2, col3 = st.columns([1,6,1])
            with col1:
                st.write("")
            with col2:
                st.image(image, width=(200))
            with col3:
                st.write("")
            st.markdown("An actinic keratosis is a rough, scaly patch on the skin that develops from years of sun exposure. Left untreated, the risk of actinic keratoses turning into a type of skin cancer called squamous cell carcinoma is about 5% to 10%. (Source: mayoclinic.org)")
            st.markdown("Bowen's disease is a very early form of skin cancer that's easily treatable. The main sign is a red, scaly patch on the skin. The patch is usually very slow growing, but there's a small chance it could turn into a more serious type of skin cancer if left untreated. (Source: nhs.uk)")
