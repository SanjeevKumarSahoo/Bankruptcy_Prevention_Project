# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 04:01:49 2023

@author: Admin
"""

import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('E:/DATA SCIENCE/Project/decision_tree.sav', 'rb'))

# Creating a function for prediction
def bankruptcy_prediction(input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 1):
      return'Non-Bankruptcy'
    else:
      return'Bankruptcy'
      


def main():
    
      
    # giving a title
    st.title('Bankruptcy prediction Web App')
    
    
    # getting the input data from the user
    
    
    number1 = st.number_input('Industrial Risk ')
    number2 = st.number_input('Management Risk')
    number3 = st.number_input('Financial Flexibility')
    number4 = st.number_input('Operating Risk')
    number5 = st.number_input('Bankruptcy Risk')
   
        
    # code for Prediction
    prevention = ''
    
    # creating a button for Prediction
    
    if st.button('Prediction Result'):
        prevention = bankruptcy_prediction([number1,number2,number3,number4,number5])
        
        
    st.success(prevention)
    
    
    
    
    
if __name__ == '__main__':
    main()











      