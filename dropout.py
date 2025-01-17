import streamlit as st
import pandas as pd
import plotly.express as px

# Title
st.title("Predicting Student Dropout and Academic Success")

# Upload CSV file
uploaded_file = st.file_uploader("Upload the dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Basic Data Overview
    st.subheader("Dataset Overview")
    st.write(df.head())

    # Select Feature for Analysis
    st.sidebar.subheader("Interactive Data Visualization")
    feature_x = st.sidebar.selectbox("Select X-axis feature", df.columns)
    feature_y = st.sidebar.selectbox("Select Y-axis feature", df.columns)
    color_feature = st.sidebar.selectbox("Select feature for color grouping", df.columns)

    # Scatter Plot
    st.subheader(f"Scatter Plot: {feature_x} vs {feature_y}")
    fig_scatter = px.scatter(df, x=feature_x, y=feature_y, color=color_feature,
                             title=f"{feature_x} vs {feature_y} by {color_feature}")
    st.plotly_chart(fig_scatter)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig_heatmap = px.imshow(corr, text_auto=True, title="Feature Correlation Heatmap")
    st.plotly_chart(fig_heatmap)

    # Dropout Rate Analysis
    if 'Target' in df.columns:
        st.subheader("Dropout Rate Analysis")
        dropout_counts = df['Target'].value_counts()
        fig_bar = px.bar(dropout_counts, x=dropout_counts.index, y=dropout_counts.values,
                         labels={'x': 'Dropout Status', 'y': 'Count'},
                         title="Distribution of Dropout Status")
        st.plotly_chart(fig_bar)
else:
    st.info("Please upload a CSV file to proceed.")
