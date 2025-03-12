import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def visualization_page():
    if st.session_state.data is None:
        st.warning("🚨 Please upload data first!")
        return
    
    st.title("📊 Visualization")
    st.markdown("### Explore your data through beautiful visualizations! ✨")
    
    data = st.session_state.data
    
    # Smart sampling for large datasets
    if data.shape[0] > 10000:
        sample_size = st.slider("Sample size for visualization", 1000, min(10000, data.shape[0]), 5000)
        data = data.sample(sample_size)
        st.info(f"🔍 Using a sample of {sample_size} rows for visualization")
    
    viz_type = st.selectbox(
        "🎨 Select visualization type",
        ["📊 Correlation Heatmap", "📈 Scatter Plot", "📦 Box Plot", "📊 Histogram", "📈 Line Plot"]
    )
    
    if viz_type == "📊 Correlation Heatmap":
        numerical_data = data.select_dtypes(include=['float64', 'int64'])
        if not numerical_data.empty:
            fig = px.imshow(numerical_data.corr(), 
                          color_continuous_scale='RdBu',
                          title='📊 Correlation Heatmap')
            st.plotly_chart(fig)
        else:
            st.warning("❌ No numerical columns available for correlation analysis")
    
    elif viz_type == "📈 Scatter Plot":
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("📈 Select X axis", numerical_cols)
        with col2:
            y_col = st.selectbox("📈 Select Y axis", numerical_cols)
        with col3:
            color_col = st.selectbox("🎨 Color by (optional)", ['None'] + list(data.columns))
        
        color = None if color_col == 'None' else data[color_col]
        fig = px.scatter(data, x=x_col, y=y_col, color=color,
                        title=f'📈 {x_col} vs {y_col}')
        st.plotly_chart(fig)
    
    elif viz_type == "📦 Box Plot":
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        selected_cols = st.multiselect("📊 Select columns for box plot", numerical_cols)
        
        if selected_cols:
            group_by = st.selectbox("🔍 Group by (optional)", ['None'] + list(data.columns))
            if group_by != 'None':
                fig = px.box(data, y=selected_cols[0], x=group_by,
                           title=f'📦 Box Plot: {", ".join(selected_cols)} by {group_by}')
            else:
                fig = px.box(data, y=selected_cols,
                           title=f'📦 Box Plot: {", ".join(selected_cols)}')
            st.plotly_chart(fig)
    
    elif viz_type == "📊 Histogram":
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        col1, col2 = st.columns(2)
        with col1:
            selected_col = st.selectbox("📊 Select column for histogram", numerical_cols)
        with col2:
            bins = st.slider("Number of bins", 5, 100, 30)
        
        fig = px.histogram(data, x=selected_col, nbins=bins,
                         title=f'📊 Histogram: {selected_col}')
        st.plotly_chart(fig)
    
    elif viz_type == "📈 Line Plot":
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("📈 Select X axis (time/sequence)", numerical_cols)
        with col2:
            y_cols = st.multiselect("📈 Select Y axis (values)", numerical_cols)
        with col3:
            group_by = st.selectbox("🔍 Group by (optional)", ['None'] + list(data.columns))
        
        if y_cols:
            if group_by != 'None':
                fig = px.line(data, x=x_col, y=y_cols, color=group_by,
                            title=f'📈 Line Plot: {", ".join(y_cols)} over {x_col}')
            else:
                fig = px.line(data, x=x_col, y=y_cols,
                            title=f'📈 Line Plot: {", ".join(y_cols)} over {x_col}')
            st.plotly_chart(fig)