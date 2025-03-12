import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from upload import data_upload_page
from preprocess import preprocessing_page
from modeltrain import model_training_page
from visualization import visualization_page


CLASSIFICATION_MODELS = {
    "🌳 Random Forest (Best for Complex Data) 🌟": RandomForestClassifier,
    "🎯 Logistic Regression (Simple & Fast) ⚡": LogisticRegression,
    "🎪 Support Vector Machine (Good for Small Datasets) 🎭": SVC,
    "🌲 Decision Tree (Easy to Interpret) 📚": DecisionTreeClassifier,
    "🚀 XGBoost (High Performance) 🏆": XGBClassifier
}

REGRESSION_MODELS = {
    "🌳 Random Forest (Robust Predictions) 🌟": RandomForestRegressor,
    "📈 Linear Regression (Simple & Fast) ⚡": LinearRegression,
    "🎪 Support Vector Regression (Complex Patterns) 🎭": SVR,
    "🌲 Decision Tree (Clear Decision Rules) 📚": DecisionTreeRegressor,
    "🚀 XGBoost (Champion Performance) 🏆": XGBRegressor
}

SCALING_METHODS = {
    "✨ StandardScaler (Best for Normal Distribution)": StandardScaler,
    "🌈 MinMaxScaler (Best for Known Bounds)": MinMaxScaler,
    "💪 RobustScaler (Best for Outliers)": RobustScaler
}


def main():
    st.set_page_config(
        page_title="✨ AI Data Wizard – Smart Insights",
        page_icon="🧙‍♂️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
        }
        .upload-section {
            border: 2px dashed #ccc;
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .preprocessing-step {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🧙‍♂️ AI Data Wizard ")
    st.sidebar.markdown("---")
    page = st.sidebar.selectbox(
        "🎯 Navigation",
        ["🔮 Welcome", "📤 Data Upload", "⚡ Preprocessing", "🚀 Model Training", "📊 Visualization"]
    )
    
    # Initialize session state
    for key in ['data', 'processed_data', 'model', 'preprocessing_steps']:
        if key not in st.session_state:
            st.session_state[key] = None
            
    if st.session_state.preprocessing_steps is None:
        st.session_state.preprocessing_steps = []
    
    # Page routing
    if page == "🔮 Welcome":
        welcome_page()
    elif page == "📤 Data Upload":
        data_upload_page()
    elif page == "⚡ Preprocessing":
        preprocessing_page()
    elif page == "🚀 Model Training":
        model_training_page()
    else:
        visualization_page()

def welcome_page():
    st.title("🔮 Welcome to AI Data Wizard ")
    
    st.markdown("""
    ### ✨ Transform Your Data into Magic!
    
    Get started with these easy steps:
    1. 📤 **Upload your dataset** in CSV format
    2. ⚡ **Preprocess your data** with our interactive tools
    3. 🚀 **Train powerful models** with just a few clicks
    4. 📊 **Visualize insights** from your data
    
    Let's begin your data science journey! 🚀
    """)
    
    # Quick start guide
    with st.expander("📚 Quick Start Guide"):
        st.markdown("""
        ### How to Use AI Data Wizard Pro
        
        1. **Data Upload**
           - Support for CSV, Excel, JSON, Parquet Files
           - Automatic data type detection
           - Basic statistics and overview
        
        2. **Preprocessing**
           - Handle missing values
           - Encode categorical variables
           - Scale numerical features
           - Remove duplicates
        
        3. **Model Training**
           - Multiple algorithms available
           - Automatic validation
           - Performance metrics
           - Downloadable models
        
        4. **Visualization**
           - Interactive plots
           - Correlation analysis
           - Distribution views
        """)

if __name__ == "__main__":
    main()