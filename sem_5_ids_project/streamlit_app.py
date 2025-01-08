import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Custom Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    # Update with your dataset loading code
    data = pd.read_csv("archive/South_Asian_dataset.csv")
    return data

data = load_data()

st.title("📊 Custom Data Analysis App")
st.write("Explore your dataset interactively.")

st.sidebar.header("Select Analysis")
options = [
    "Overview",
    "Analysis 1",
    "Analysis 2",
    "Custom Filters",
]
selected_option = st.sidebar.radio("Choose an analysis:", options)

if selected_option == "Overview":
    st.header("📋 Dataset Overview")
    st.write(data.head())  # Display the first few rows
    st.metric(label="Total Rows", value=len(data))
    st.metric(label="Total Columns", value=len(data.columns))

elif selected_option == "Analysis 1":
    st.header("📈 Analysis 1")
    # Example visualization
    fig = px.histogram(data, x="Column1", title="Histogram of Column1")
    st.plotly_chart(fig)

elif selected_option == "Analysis 2":
    st.header("📉 Analysis 2")
    # Another example
    fig = px.scatter(data, x="ColumnX", y="ColumnY", color="CategoryColumn")
    st.plotly_chart(fig)

elif selected_option == "Custom Filters":
    st.header("🔍 Custom Filters")
    selected_value = st.selectbox("Filter by ColumnX:", options=data["ColumnX"].unique())
    filtered_data = data[data["ColumnX"] == selected_value]
    st.write(filtered_data)

st.markdown(
    """
    <footer style='text-align:center; margin-top:20px;'>
        <hr>
        <p>📊 Custom Analysis App - Created by Abiha</p>
    </footer>
    """,
    unsafe_allow_html=True
)
