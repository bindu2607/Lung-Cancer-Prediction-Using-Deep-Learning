import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Define the mapping from class index to human-readable label
label_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}

# Load the saved Keras model
model = load_model('my_model.h5')

# Load the saved scaler object
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Example new data (make sure to provide exactly 23 features as in your training)
# Replace my_listX with your actual input features
my_list0 = [33,1,2,4,5,4,3,2,2,4,3,2,2,4,3,4,2,2,3,1,2,3,4]
my_list1 = [17,1,3,1,5,3,4,2,2,2,2,4,2,3,1,3,7,8,6,2,1,7,2]
my_list2 = [64,2,6,8,7,7,7,6,7,7,7,8,7,7,9,6,5,7,2,4,3,1,4]

# Convert your chosen input list to numpy array, shape (1, 23)
new_data = np.array([my_list2])  # For example, using my_list2

# Scale the new data using the loaded scaler
new_data_scaled = scaler.transform(new_data)

# Predict class probabilities
predictions = model.predict(new_data_scaled)

# Get predicted class index
predicted_class_index = np.argmax(predictions, axis=1)[0]

# Get predicted class label
predicted_class_label = label_mapping[predicted_class_index]

# Print results with formatted probabilities
print(f"Predicted Class Index: {predicted_class_index}")
print(f"Predicted Risk Level: {predicted_class_label}")


