import streamlit as st
import tensorflow as tf
import pickle
import os
from keras.models import load_model
import numpy as np

from streamlit_option_menu import option_menu

heart_disease = pickle.load(open('heartdisease.sav','rb'))
liver_disease = pickle.load(open('liverdisease.sav','rb'))

#new_model = tf.keras.models.load_model('asthmadisease_r.h5')

with st.sidebar:
    selected = option_menu('Multiple disease Prediction System',
                           ['Kidney Disease Prediction',
                            'Heart Disease Prediction',
                            'Liver Disease Prediction'],
                           default_index = 0)

#kidney Disease Prediction
if selected == 'Kidney Disease Prediction':
    
    st.header('Kidney Disease Prediction')
    Class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

    model = load_model('kidney_disease_classifier_r.h5')

    def classify_images(image_path):
        input_image = tf.keras.utils.load_img(image_path, target_size=(255,255))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array,0)

        predictions = model.predict(input_image_exp_dim)
        score = tf.nn.softmax(predictions[0])
        outcome = 'This image most likely belongs to '+Class_names[np.argmax(score)]+' with a '+str(100 * np.max(score))+' percent confidence.'
        return outcome

    uploaded_file = st.file_uploader('Upload an Image')
    if uploaded_file is not None:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(uploaded_file, width = 200)

        st.markdown(classify_images(uploaded_file))
        
        
# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex : 0 -> Male , 1 -> Female')

    with col3:
        cp = st.text_input('Chest Pain types : 0 - ASY , 1 - NAP ,2 - ATA ,3 - TA')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure : Minimum - 0')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl : Minimum - 0')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results : 0 - Normal , 1 - LVH , 2 - ST')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved : Minimum - 60')

    with col3:
        exang = st.text_input('Exercise Induced Angina : 0 - No , 1 - Yes')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise : Minimum -> -2.6')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment : 0 - Flat , 1 - Up , 2 - Down')


    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'This person is having heart disease'
        else:
            heart_diagnosis = 'This person does not have any heart disease'

    st.success(heart_diagnosis)
    
#Liver Disease
if selected == 'Liver Disease Prediction':

    # page title
    st.title('Liver Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        gender = st.text_input('Gender : 0 -> Male , 1 -> Female')

    with col3:
        bmi = st.text_input('BMI : Range: 15 to 40')

    with col1:
        alcom = st.text_input('Alchol Consumption : Range: 0 to 20 units per week')

    with col2:
        smoke = st.text_input('Smoking : No -> 0 or Yes -> 1 ')

    with col3:
        gs = st.text_input('Genetic Risk : Low -> 0, Medium -> 1, High -> 2')

    with col1:
        phyAc = st.text_input('Physical Activity : Range: 0 to 10 hours per week')

    with col2:
        dia = st.text_input('Diabetes : No -> 0 or Yes -> 1')

    with col3:
        hypTen = st.text_input('Hypertension : No -> 0 or Yes -> 1')

    with col1:
        livertest = st.text_input('LiverFunctionTest : Range: 20 to 100')


    # code for Prediction
    Liver_diagnosis = ''

    # creating a button for Prediction
    if st.button('Liver Disease Test Result'):

        user_input = [age, gender ,bmi ,alcom ,smoke ,gs ,phyAc,dia ,hypTen ,livertest]

        user_input = [float(x) for x in user_input]

        liver_prediction = liver_disease.predict([user_input])

        if liver_prediction[0] == 1:
            Liver_diagnosis = 'This person is having Liver disease'
        else:
            Liver_diagnosis = 'This person does not have any Liver disease'

    st.success(Liver_diagnosis)