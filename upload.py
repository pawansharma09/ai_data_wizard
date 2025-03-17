import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def data_upload_page():
    st.title("📤 Data Upload")

    # Styled upload section with animated arrows
    st.markdown("""
        <div style="text-align: center;">
            <h3>🔮 Upload Your Dataset</h3>
            <p>Supported formats: CSV, Excel, JSON, Parquet</p>
            <div style="font-size: 2rem; margin: 1rem;">
                ⬇️⬇️⬇️
            </div>
        </div>
    """, unsafe_allow_html=True)

    # File uploader (supports multiple formats)
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls', 'json', 'parquet'], key="file_uploader")

    if uploaded_file:
        try:
            # Progress bar for file processing
            with st.spinner("⏳ Processing your file..."):
                # Detecting file type and loading data accordingly
                file_extension = uploaded_file.name.split('.')[-1].lower()

                # Display file type icon dynamically
                file_icons = {
                    'csv': '📄',
                    'xlsx': '📊',
                    'xls': '📊',
                    'json': '📑',
                    'parquet': '📂'
                }
                file_icon = file_icons.get(file_extension, '📁')

                st.markdown(f"""
                    <div style="text-align: center; margin: 1rem;">
                        <h3>{file_icon} Uploaded File: {uploaded_file.name}</h3>
                    </div>
                """, unsafe_allow_html=True)

                if file_extension == 'csv':
                    data = pd.read_csv(uploaded_file)
                elif file_extension in ['xlsx', 'xls']:
                    data = pd.read_excel(uploaded_file)
                elif file_extension == 'json':
                    data = pd.read_json(uploaded_file)
                elif file_extension == 'parquet':
                    data = pd.read_parquet(uploaded_file)
                else:
                    st.error("❌ Unsupported file format!")
                    return

                st.session_state.data = data  # Store dataset in session state

                # Success message with dataset summary
                st.success("🎉 Dataset uploaded successfully!")
                st.snow()  # Add celebratory balloons animation

                # Display dataset metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📝 Rows", data.shape[0])
                with col2:
                    st.metric("📋 Columns", data.shape[1])
                with col3:
                    st.metric("❌ Missing Values", data.isna().sum().sum())

                # Data preview
                st.subheader("👀 Data Preview")
                st.dataframe(data.head())

                # Data type and memory usage details
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("📊 Data Types")
                    dtypes_info = pd.DataFrame(data.dtypes, columns=["Type"])
                    dtypes_info["Unique Count"] = data.nunique()
                    st.write(dtypes_info)

                with col2:
                    st.subheader("📈 Memory Usage \n (Per Column)")

                    # Convert memory usage to KB/MB for readability
                    memory_usage = data.memory_usage(deep=True)
                    memory_usage_readable = memory_usage.apply(lambda x: f"{x / 1024:.2f} KB" if x < 1024**2 else f"{x / 1024**2:.2f} MB")

                    # Display as dataframe
                    mem_usage_df = pd.DataFrame({"Column": memory_usage.index, "Memory Usage": memory_usage_readable}).iloc[1:]
                    st.dataframe(mem_usage_df, hide_index=True)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
