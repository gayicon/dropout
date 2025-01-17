import streamlit as st
import pandas as pd
import plotly.express as px

# Title
st.title("Student Dropout and Academic Success Analysis")

# Load Dataset from GitHub URL
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/gayicon/dropout/main/data.csv"
    return pd.read_csv(url)

df = load_data()

# Dataset Overview
st.subheader("Dataset Overview")
st.write(df.head())

# Sidebar for Feature Selection
st.sidebar.subheader("Interactive Visualization Settings")
feature_x = st.sidebar.selectbox("Select X-axis feature", df.columns)
feature_y = st.sidebar.selectbox("Select Y-axis feature", df.columns)
color_feature = st.sidebar.selectbox("Select feature for color grouping", df.columns)

# Scatter Plot
st.subheader(f"Scatter Plot: {feature_x} vs {feature_y}")
fig_scatter = px.scatter(df, x=feature_x, y=feature_y, color=color_feature,
                         title=f"{feature_x} vs {feature_y} grouped by {color_feature}")
st.plotly_chart(fig_scatter)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
corr = df.corr(numeric_only=True)
fig_heatmap = px.imshow(corr, text_auto=True, title="Feature Correlation Heatmap")
st.plotly_chart(fig_heatmap)

# Dropout Analysis (if 'Target' column exists)
if 'Target' in df.columns:
    st.subheader("Dropout Status Distribution")
    dropout_counts = df['Target'].value_counts()
    fig_bar = px.bar(x=dropout_counts.index, y=dropout_counts.values,
                     labels={'x': 'Dropout Status', 'y': 'Count'},
                     title="Dropout Status Count")
    st.plotly_chart(fig_bar)
else:
    st.info("'Target' column not found in the dataset for dropout analysis.")
