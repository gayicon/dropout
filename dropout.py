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

# Clean column names: strip extra spaces
df.columns = df.columns.str.strip()

# Dataset Overview
st.subheader("Dataset Overview")
st.write(df.head())

# Sidebar for Feature Selection
st.sidebar.subheader("Interactive Visualization Settings")

# Filter out unnecessary columns (add to this list as needed)
filter_columns = [
    'Marital status', 'Application mode', 'Course', 'Gender', 'Age at enrollment', 
    'Admission grade', 'Curricular units 1st sem (credited)', 'Curricular units 1st sem (approved)', 
    'Curricular units 2nd sem (credited)', 'Target', 'Unemployment rate', 'Inflation rate', 'GDP'
]

# Ensure that only relevant columns appear in the sidebar
available_columns = [col for col in df.columns if col in filter_columns]

feature_x = st.sidebar.selectbox("Select X-axis feature", available_columns)
feature_y = st.sidebar.selectbox("Select Y-axis feature", available_columns)
color_feature = st.sidebar.selectbox("Select feature for color grouping", available_columns)

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
