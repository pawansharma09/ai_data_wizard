import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler

# Define scaling methods
SCALING_METHODS = {
    "✨ StandardScaler (Best for Normal Distribution)": StandardScaler,
    "🌈 MinMaxScaler (Best for Known Bounds)": MinMaxScaler,
    "💪 RobustScaler (Best for Outliers)": RobustScaler
}

def preprocessing_page():
    if st.session_state.data is None:
        st.warning("🚨 Please upload data first!")
        return
    
    st.title("⚡ Data Preprocessing")
    data = st.session_state.data.copy()
    
    # Preprocessing steps container
    st.markdown("### 🔧 Preprocessing Steps")
    
    # 1. Handle Missing Values
    st.subheader("1️⃣ Handle Missing Values")
    missing_cols = data.columns[data.isnull().any()].tolist()
    if missing_cols:
        st.write("📊 Columns with missing values:")
        for col in missing_cols:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                strategy = st.selectbox(
                    f"Strategy for {col}",
                    ['None', 'Drop', 'Mean', 'Median', 'Mode', 'Zero'],
                    key=f"missing_{col}"
                )
            with col2:
                st.metric("Missing Count", data[col].isnull().sum())
            with col3:
                st.metric("Missing %", f"{(data[col].isnull().sum() / len(data) * 100):.1f}%")
            
            if strategy != 'None':
                if strategy == 'Drop':
                    data = data.dropna(subset=[col])
                elif strategy == 'Mean':
                    data[col] = data[col].fillna(data[col].mean())
                elif strategy == 'Median':
                    data[col] = data[col].fillna(data[col].median())
                elif strategy == 'Mode':
                    data[col] = data[col].fillna(data[col].mode()[0])
                elif strategy == 'Zero':
                    data[col] = data[col].fillna(0)
    else:
        st.info("✨ No missing values found!")
    
    # 2. Feature Scaling
    st.subheader("2️⃣ Feature Scaling")
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    
    col1, col2 = st.columns([1, 2])
    with col1:
        scaler_method = st.selectbox(
            "Select scaling method",
            ['None'] + list(SCALING_METHODS.keys())
        )
    with col2:
        if scaler_method != 'None':
            scale_cols = st.multiselect(
                "Select columns to scale",
                numerical_cols
            )
            if scale_cols:
                scaler = SCALING_METHODS[scaler_method]()
                data[scale_cols] = scaler.fit_transform(data[scale_cols])
    
    # 3. Encoding
    st.subheader("3️⃣ Categorical Encoding")
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        st.write("🎨 Categorical columns detected:")
        for col in categorical_cols:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                encoding = st.selectbox(
                    f"Encode {col}",
                    ['None', 'Label Encoding', 'One-Hot Encoding'],
                    key=f"encode_{col}"
                )
            with col2:
                st.metric("Unique Values", data[col].nunique())
            with col3:
                st.metric("Top Value", data[col].mode()[0])
            
            if encoding == 'Label Encoding':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
            elif encoding == 'One-Hot Encoding':
                data = pd.get_dummies(data, columns=[col])
    else:
        st.info("✨ No Categorical Column found!")
    
    # 4. Remove Duplicates
    st.subheader("4️⃣ Remove Duplicates")
    dup_count = data.duplicated().sum()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Duplicate Rows", dup_count)
    with col2:
        if dup_count > 0:
            remove_dups = st.checkbox("Remove duplicate rows")
            if remove_dups:
                data = data.drop_duplicates()
    
    # Apply preprocessing
    if st.button("⚡ Apply Preprocessing"):
        st.session_state.processed_data = data
        st.success(f"🎉 Preprocessing completed! Shape: {data.shape}")
        
        # Show sample of processed data
        st.subheader("🔍 Processed Data Preview")
        st.dataframe(data.head())
        
        # Show changes summary
        st.subheader("📊 Changes Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Shape", f"{st.session_state.data.shape}")
        with col2:
            st.metric("Processed Shape", f"{data.shape}")
