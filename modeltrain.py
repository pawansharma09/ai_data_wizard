import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import pickle
import base64

# Define model dictionaries
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

def create_download_button(model, filename="model.pkl"):
    """Create a styled download button for the model"""
    buffer = BytesIO()
    pickle.dump(model, buffer)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    
    button_style = """
        <style>
        .download-button {
            background-color: black;
            border: none;
            color: white;
            padding: 12px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
            transition: background-color 0.3s;
        }
        .download-button:hover {
            background-color: black;
        }
        </style>
    """
    
    button_html = f"""
        {button_style}
        <a href="data:application/octet-stream;base64,{b64}" 
           download="{filename}" 
           class="download-button">
           ⬇️ Download Trained Model
        </a>
    """
    return button_html

def model_training_page():
    if st.session_state.processed_data is None:
        st.warning("🚨 Please preprocess your data first!")
        return
    
    st.title("🚀 Model Training")
    data = st.session_state.processed_data
    
    # Model selection interface
    st.markdown("### 🤖 Select Your Model")
    
    col1, col2 = st.columns(2)
    with col1:
        problem_type = st.selectbox(
            "🎯 Problem Type",
            ["Classification", "Regression"],
            help="Select the type of machine learning problem"
        )
    with col2:
        model_dict = CLASSIFICATION_MODELS if problem_type == "Classification" else REGRESSION_MODELS
        selected_model = st.selectbox("🔮 Select Model", list(model_dict.keys()))
    
    # Target selection
    st.markdown("### 🎯 Select Target Variable")
    target_col = st.selectbox(
        "Choose the target variable",
        data.columns,
        help="This is the variable you want to predict"
    )
    
    # Training configuration
    st.markdown("### ⚙️ Training Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2)
    with col2:
        cv_folds = st.slider("Cross-validation Folds", 2, 10, 5)
    with col3:
        random_state = st.number_input("Random State", value=42)
    
    if st.button("🚀 Train Model"):
        with st.spinner("🔮 Training in progress..."):
            # Prepare data
            X = data.drop(target_col, axis=1)
            y = data[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Initialize and train model
            model = model_dict[selected_model]()
            model.fit(X_train, y_train)
            st.session_state.model = model
            
            # Evaluate model
            y_pred = model.predict(X_test)
            
            # Display results in a nice format
            st.markdown("### 📊 Model Performance")
            
            if problem_type == "Classification":
                # Classification metrics
                accuracy = accuracy_score(y_test, y_pred)
                cv_scores = cross_val_score(model, X, y, cv=cv_folds)
                
                # Display metrics in cards
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                        <div class="metric-card">
                            <h4 style="color: black">🎯 Test Accuracy</h4>
                            <h2 style="color: #4CAF50">{:.2%}</h2>
                        </div>
                    """.format(accuracy), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                        <div class="metric-card">
                            <h4 style="color: black">🔄 Cross-validation Score</h4>
                            <h2 style="color: #4CAF50">{:.2%} ± {:.2%}</h2>
                        </div>
                    """.format(cv_scores.mean(), cv_scores.std()*2), unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                        <div class="metric-card">
                            <h4 style="color: black">📈 Model Type</h4>
                            <h2 style="color: #4CAF50">Classification</h2>
                        </div>
                    """.format(), unsafe_allow_html=True)
                
                # Classification report
                st.markdown("### 📋 Detailed Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0))
                
            else:
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = model.score(X_test, y_test)
                
                # Display metrics in cards
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                        <div class="metric-card">
                            <h4 style="color: black">📉 RMSE</h4>
                            <h2 style="color: #4CAF50">{:.4f}</h2>
                        </div>
                    """.format(rmse), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                        <div class="metric-card">
                            <h4 style="color: black">📈 R² Score</h4>
                            <h2 style="color: #4CAF50">{:.4f}</h2>
                        </div>
                    """.format(r2), unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                        <div class="metric-card">
                            <h4 style="color: black">📈 Model Type</h4>
                            <h2 style="color: #4CAF50">Regression</h2>
                        </div>
                    """.format(), unsafe_allow_html=True)
            
            # Feature importance plot
            if hasattr(model, 'feature_importances_'):
                st.markdown("### 🔍 Feature Importance Analysis")
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig = px.bar(feature_importance, 
                            x='feature', 
                            y='importance',
                            title='🎯 Feature Importance Analysis')
                fig.update_layout(
                    xaxis_title="Features",
                    yaxis_title="Importance Score",
                    showlegend=False
                )
                st.plotly_chart(fig)
            
            # Predictions vs Actual plot
            st.markdown("### 📈 Predictions vs Actual Values")
            plot_data = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred
            })
            
            fig = px.scatter(plot_data, 
                           x='Actual', 
                           y='Predicted',
                           title='🎯 Predictions vs Actual Values')
            fig.add_trace(
                go.Scatter(x=[plot_data.Actual.min(), plot_data.Actual.max()],
                          y=[plot_data.Actual.min(), plot_data.Actual.max()],
                          mode='lines',
                          name='Perfect Prediction',
                          line=dict(color='red', dash='dash'))
            )
            st.plotly_chart(fig)
            
            # Model download section
            st.markdown("### ⬇️ Download Trained Model")
            st.markdown(create_download_button(model), unsafe_allow_html=True)
            
            # Model summary
            st.markdown("### 📋 Model Summary")
            st.json({
                "Model Type": selected_model,
                "Number of Features": X.shape[1],
                "Training Set Size": X_train.shape[0],
                "Test Set Size": X_test.shape[0],
                "Cross-validation Folds": cv_folds,
                "Random State": random_state
            })