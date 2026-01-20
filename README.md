# 🎯 Customer Churn Prediction System

An end-to-end machine learning project using Artificial Neural Networks (ANN) to predict customer churn for a telecommunications company, with an interactive Streamlit web application.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [File Structure](#file-structure)

## 🎯 Project Overview

This project predicts whether a customer will churn (leave the company) based on their demographics, account information, and service usage patterns. The system uses a deep learning ANN model trained on the Telco Customer Churn dataset.

## ✨ Features

- **Deep Learning Model**: Multi-layer ANN with dropout and batch normalization
- **Interactive Web App**: User-friendly Streamlit interface
- **Real-time Predictions**: Instant churn probability calculations
- **Visualization Dashboard**: Performance metrics and feature importance charts
- **Risk Analysis**: Automated risk factor identification
- **Actionable Insights**: Personalized recommendations for customer retention

## 🛠️ Technology Stack

- **Python 3.8+**
- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Preprocessing and evaluation
- **Plotly**: Interactive visualizations
- **Matplotlib & Seaborn**: Static visualizations

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
tensorflow==2.13.0
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.16.1
```

## 📊 Dataset

### Download the Dataset

1. Download from Kaggle: [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
2. Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root directory

### Dataset Features

- **Demographics**: Gender, Senior Citizen, Partner, Dependents
- **Account Info**: Tenure, Contract Type, Payment Method, Billing
- **Services**: Phone, Internet, Security, Backup, Streaming, etc.
- **Charges**: Monthly Charges, Total Charges
- **Target**: Churn (Yes/No)

**Dataset Size**: 7,043 customers with 21 features

## 🚀 Usage

### Step 1: Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the data
- Train the ANN model
- Save model files (`.h5`, `.pkl`)
- Generate performance visualizations

**Expected Output:**
- `churn_prediction_model.h5` - Trained model
- `scaler.pkl` - Feature scaler
- `label_encoders.pkl` - Categorical encoders
- `feature_names.pkl` - Feature list
- `model_performance.png` - Training curves
- `confusion_matrix.png` - Confusion matrix heatmap

### Step 2: Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Step 3: Make Predictions

1. Fill in customer information in the sidebar
2. Click "Predict Churn" button
3. View prediction results and recommendations

## 🧠 Model Architecture

```
Input Layer (24 features)
    ↓
Dense Layer (128 neurons, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (0.3)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (0.3)
    ↓
Dense Layer (32 neurons, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (0.2)
    ↓
Dense Layer (16 neurons, ReLU)
    ↓
Dropout (0.2)
    ↓
Output Layer (1 neuron, Sigmoid)
```

**Total Parameters**: ~12,000
**Optimizer**: Adam
**Loss Function**: Binary Crossentropy
**Metrics**: Accuracy, AUC

## 📈 Results

### Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 82% |
| **Precision** | 79% |
| **Recall** | 76% |
| **F1-Score** | 77% |
| **ROC-AUC** | 85% |

### Feature Importance (Top 6)

1. **Tenure** (85%) - Longer tenure = lower churn
2. **Contract Type** (78%) - Long-term contracts reduce churn
3. **Monthly Charges** (72%) - Higher charges increase churn
4. **Tech Support** (65%) - Support reduces churn
5. **Payment Method** (58%) - Auto payments reduce churn
6. **Online Security** (52%) - Security services reduce churn

## 📁 File Structure

```
customer-churn-prediction/
│
├── train_model.py              # Model training script
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
│
├── churn_prediction_model.h5   # Trained model
├── scaler.pkl                  # Feature scaler
├── label_encoders.pkl          # Encoders
├── feature_names.pkl           # Feature list
│
├── model_performance.png       # Training visualizations
└── confusion_matrix.png        # Confusion matrix
```

## 🎨 Streamlit App Features

### Dashboard Components

1. **Input Form** (Sidebar)
   - Customer demographics
   - Account information
   - Service details
   - Billing information

2. **Prediction Results**
   - Churn probability gauge
   - Risk level indicator
   - Prediction status

3. **Visualizations**
   - Interactive gauge chart
   - Risk factors bar chart
   - Performance metrics
   - Feature importance

4. **Recommendations**
   - Personalized retention strategies
   - Action items based on risk level

## 🔧 Customization

### Modify Model Architecture

Edit the `create_model()` function in `train_model.py`:

```python
def create_model(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),  # Increase neurons
        # Add more layers...
    ])
    return model
```

### Adjust Prediction Threshold

In `app.py`, modify the threshold:

```python
churn_prediction = 1 if churn_probability > 0.4 else 0  # Changed from 0.5
```

### Change App Styling

Modify the CSS in `app.py`:

```python
st.markdown("""
<style>
    .main-header {
        color: #your-color;
    }
</style>
""", unsafe_allow_html=True)
```

## 📊 Model Training Tips

### Improve Performance

1. **Hyperparameter Tuning**:
   - Adjust learning rate
   - Change batch size
   - Modify dropout rates

2. **Feature Engineering**:
   - Create interaction features
   - Apply polynomial features
   - Use feature selection

3. **Handle Imbalance**:
   - Use SMOTE for oversampling
   - Apply class weights
   - Try different sampling strategies

### Example Training Configuration

```python
# In train_model.py
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=150,              # Increase epochs
    batch_size=64,           # Adjust batch size
    class_weight={0: 1, 1: 2}  # Handle imbalance
)
```

## 🐛 Troubleshooting

### Common Issues

**Issue 1: Model files not found**
```
Solution: Run train_model.py first to generate model files
```

**Issue 2: Package compatibility errors**
```
Solution: Install exact versions from requirements.txt
pip install -r requirements.txt --force-reinstall
```

**Issue 3: Streamlit not loading**
```
Solution: Check if port 8501 is available
streamlit run app.py --server.port 8502
```

## 📝 Future Enhancements

- [ ] Add batch prediction from CSV upload
- [ ] Implement model versioning
- [ ] Add A/B testing capabilities
- [ ] Create API endpoint with FastAPI
- [ ] Deploy on cloud (AWS/Azure/GCP)
- [ ] Add user authentication
- [ ] Implement model monitoring dashboard
- [ ] Create mobile-responsive design

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Shehryar Naveed**
- GitHub: [@](https://github.com/shery0099x)

## 🙏 Acknowledgments

- Dataset: [IBM Sample Data Sets](https://www.kaggle.com/blastchar/telco-customer-churn)
- TensorFlow Documentation
- Streamlit Community
- Kaggle Community


---

**⭐ If you find this project helpful, please give it a star!**
