# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# Loading the saved file
loaded_model = pickle.load(open('E:/DATA SCIENCE/Project/decision_tree.sav', 'rb'))

input_data = (0.5,1.0,0,0.5,0)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 1):
  print('Non-Bankruptcy')
else:
  print('Bankruptcy')
  
