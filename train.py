import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# =====================
# Load and preprocess data
# =====================

# Load dataset
df = pd.read_csv('cancer_patient_datasets.csv')

# Map target labels (Low, Medium, High) to (0, 1, 2)
level_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df['Level'] = df['Level'].replace(level_mapping)

# Save updated CSV (optional but good for consistency)
df.to_csv('cancer_patient_datasets.csv', index=False)

# Extract features (X) and labels (y)
X = df.iloc[:, 2:-1].values
y = df.iloc[:, -1].values

# One-hot encode labels (for multiclass classification)
y = to_categorical(y, num_classes=3)

# Standardize feature values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split (stratified for class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# =====================
# Define and train model
# =====================

# Create MLP model
model = Sequential([
    Input(shape=(X.shape[1],), name="input_layer"),
    Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.4),
    Dense(8, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.4),
    Dense(3, activation='softmax')  # Output layer for 3 classes
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=2
)

# Evaluate on test set
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Train Accuracy: {train_acc * 100:.2f}%")
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

# =====================
# Visualization
# =====================

# Accuracy plot
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_plot.png')
plt.show()

# Loss plot
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_plot.png')
plt.show()

# =====================
# Confusion Matrix and Classification Report
# =====================

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=["Low", "Medium", "High"]))

# =====================
# Save Model and Scaler
# =====================

# Save trained model (HDF5 format)
model.save('my_model.h5')

# Save the scaler for later use (e.g., in prediction app)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n🎉 Model and scaler saved successfully.")