# Customer Churn Prediction using Artificial Neural Networks
# Dataset: Telco Customer Churn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
import tensorflow as tf
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import pickle
import warnings
warnings.filterwarnings('ignore')
import pickle
import warnings
warnings.filterwarnings('ignore')

# =============================================
# 1. LOAD AND EXPLORE DATA
# =============================================

# Load dataset (download from: https://www.kaggle.com/blastchar/telco-customer-churn)
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nChurn Distribution:")
print(df['Churn'].value_counts())

# =============================================
# 2. DATA PREPROCESSING
# =============================================

# Drop customerID as it's not useful for prediction
df = df.drop('customerID', axis=1)

# Handle TotalCharges - convert to numeric and handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Convert SeniorCitizen to object for consistency
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

# Separate numerical and categorical columns
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Churn')  # Remove target variable

print("\nNumerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target variable
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# =============================================
# 3. FEATURE ENGINEERING
# =============================================

# Create additional features
df['ChargePerTenure'] = df['TotalCharges'] / (df['tenure'] + 1)  # Avoid division by zero
df['AvgMonthlyCharges'] = df['TotalCharges'] / (df['tenure'] + 1)

# =============================================
# 4. SPLIT DATA
# =============================================

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Churn rate in training set: {y_train.mean():.2%}")
print(f"Churn rate in test set: {y_test.mean():.2%}")

# =============================================
# 5. FEATURE SCALING
# =============================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================
# 6. BUILD ANN MODEL
# =============================================

def create_model(input_dim):
    model = Sequential([
        # Input Layer
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden Layer 1
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden Layer 2
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Hidden Layer 3
        Dense(16, activation='relu'),
        Dropout(0.2),
        
        # Output Layer
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

# Create model
model = create_model(X_train_scaled.shape[1])
print("\nModel Architecture:")
model.summary()

# =============================================
# 7. TRAIN MODEL
# =============================================

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# Train model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# =============================================
# 8. EVALUATE MODEL
# =============================================

# Predictions
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =============================================
# 9. VISUALIZATIONS
# =============================================

# Plot training history
plt.figure(figsize=(15, 5))

# Accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# ROC Curve
plt.subplot(1, 3, 3)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================
# 10. SAVE MODEL AND PREPROCESSING OBJECTS
# =============================================

# Save the trained model
model.save('churn_prediction_model.h5')
print("\nModel saved as 'churn_prediction_model.h5'")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved as 'scaler.pkl'")

# Save label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("Label encoders saved as 'label_encoders.pkl'")

# Save feature names
feature_names = X_train.columns.tolist()
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("Feature names saved as 'feature_names.pkl'")

print("\n" + "="*50)
print("Training Complete! All files saved successfully.")
print("="*50)