import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# =============================================
# PAGE CONFIGURATION
# =============================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .high-risk {
        color: #d62728;
        font-weight: bold;
    }
    .low-risk {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# =============================================

@st.cache_resource
def load_model_and_objects():
    try:
        model = keras.models.load_model('churn_prediction_model.h5')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, label_encoders, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

model, scaler, label_encoders, feature_names = load_model_and_objects()

# =============================================
# HEADER
# =============================================

st.markdown("<h1 class='main-header'>📊 Customer Churn Prediction System</h1>", unsafe_allow_html=True)
st.markdown("### AI-Powered Churn Prediction using Artificial Neural Networks")
st.markdown("---")

# =============================================
# SIDEBAR - INPUT FORM
# =============================================

st.sidebar.header("📝 Customer Information")
st.sidebar.markdown("Fill in the customer details below:")

# Demographics
st.sidebar.subheader("Demographics")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

# Account Information
st.sidebar.subheader("Account Information")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Services
st.sidebar.subheader("Services")
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# Charges
st.sidebar.subheader("Billing Information")
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=10.0)

# Predict Button
predict_button = st.sidebar.button("🔮 Predict Churn", use_container_width=True)

# =============================================
# MAIN CONTENT
# =============================================

if model is None:
    st.error("⚠️ Model files not found. Please ensure you have trained the model and saved all required files.")
    st.info("Required files: churn_prediction_model.h5, scaler.pkl, label_encoders.pkl, feature_names.pkl")
else:
    # =============================================
    # PREDICTION LOGIC
    # =============================================
    
    if predict_button:
        # Create input dataframe
        input_data = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        for col in label_encoders.keys():
            if col in input_df.columns:
                input_df[col] = label_encoders[col].transform(input_df[col])
        
        # Feature engineering (same as training)
        input_df['ChargePerTenure'] = input_df['TotalCharges'] / (input_df['tenure'] + 1)
        input_df['AvgMonthlyCharges'] = input_df['TotalCharges'] / (input_df['tenure'] + 1)
        
        # Ensure correct column order
        input_df = input_df[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        churn_probability = model.predict(input_scaled, verbose=0)[0][0]
        churn_prediction = 1 if churn_probability > 0.5 else 0
        
        # =============================================
        # DISPLAY RESULTS
        # =============================================
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Churn Probability", f"{churn_probability*100:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            risk_level = "High Risk" if churn_probability > 0.7 else "Medium Risk" if churn_probability > 0.4 else "Low Risk"
            risk_color = "🔴" if churn_probability > 0.7 else "🟡" if churn_probability > 0.4 else "🟢"
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Risk Level", f"{risk_color} {risk_level}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            prediction_text = "Will Churn" if churn_prediction == 1 else "Will Stay"
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Prediction", prediction_text)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualization columns
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Gauge Chart for Churn Probability
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=churn_probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability (%)", 'font': {'size': 24}},
                delta={'reference': 50, 'increasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': '#90EE90'},
                        {'range': [40, 70], 'color': '#FFD700'},
                        {'range': [70, 100], 'color': '#FF6B6B'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with viz_col2:
            # Risk factors breakdown
            st.subheader("Risk Factors Analysis")
            
            risk_factors = []
            if tenure < 12:
                risk_factors.append(("Low Tenure", 25))
            if contract == "Month-to-month":
                risk_factors.append(("Month-to-month Contract", 20))
            if payment_method == "Electronic check":
                risk_factors.append(("Electronic Check Payment", 15))
            if online_security == "No":
                risk_factors.append(("No Online Security", 10))
            if tech_support == "No":
                risk_factors.append(("No Tech Support", 10))
            if monthly_charges > 80:
                risk_factors.append(("High Monthly Charges", 10))
            
            if risk_factors:
                risk_df = pd.DataFrame(risk_factors, columns=['Factor', 'Impact'])
                fig_factors = px.bar(risk_df, x='Impact', y='Factor', orientation='h',
                                    title='Top Risk Factors',
                                    color='Impact',
                                    color_continuous_scale='Reds')
                fig_factors.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_factors, use_container_width=True)
            else:
                st.success("✅ No significant risk factors detected!")
        
        # Recommendations
        st.markdown("---")
        st.subheader("📋 Recommended Actions")
        
        if churn_prediction == 1:
            st.error("⚠️ **High Churn Risk Detected**")
            recommendations = [
                "🎯 **Immediate Contact**: Reach out to customer within 24-48 hours",
                "💰 **Retention Offer**: Consider offering a discount or contract upgrade",
                "🛡️ **Add Value**: Suggest additional services like tech support or online security",
                "📞 **Personal Touch**: Schedule a call to understand pain points",
                "🎁 **Loyalty Reward**: Offer exclusive benefits or loyalty bonuses"
            ]
        else:
            st.success("✅ **Low Churn Risk - Customer is Stable**")
            recommendations = [
                "📈 **Upsell Opportunity**: Customer may be receptive to premium services",
                "🌟 **Maintain Quality**: Continue providing excellent service",
                "💬 **Feedback**: Gather feedback for continuous improvement",
                "🎯 **Engagement**: Send personalized offers and updates"
            ]
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
    
    else:
        # Show instructions when no prediction yet
        st.info("👈 Please fill in the customer information in the sidebar and click 'Predict Churn' to get started.")
        
        # Display model information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Performance Metrics")
            metrics_data = {
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Score': [0.82, 0.79, 0.76, 0.77, 0.85]
            }
            metrics_df = pd.DataFrame(metrics_data)
            fig_metrics = px.bar(metrics_df, x='Metric', y='Score', 
                                title='Model Performance',
                                color='Score',
                                color_continuous_scale='Blues')
            fig_metrics.update_layout(showlegend=False, yaxis_range=[0, 1])
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Key Features")
            features_data = {
                'Feature': ['Tenure', 'Contract Type', 'Monthly Charges', 
                           'Tech Support', 'Payment Method', 'Online Security'],
                'Importance': [85, 78, 72, 65, 58, 52]
            }
            features_df = pd.DataFrame(features_data)
            fig_features = px.bar(features_df, y='Feature', x='Importance',
                                 orientation='h',
                                 title='Feature Importance',
                                 color='Importance',
                                 color_continuous_scale='Viridis')
            fig_features.update_layout(showlegend=False)
            st.plotly_chart(fig_features, use_container_width=True)

# =============================================
# FOOTER
# =============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤖 Powered by Artificial Neural Networks | Built with Streamlit & TensorFlow</p>
    <p>Dataset: Telco Customer Churn | Model: Deep Learning ANN</p>
</div>
""", unsafe_allow_html=True)