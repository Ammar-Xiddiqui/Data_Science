import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.set_page_config(
    page_title="South Asian Countries Analysis",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    # Load your dataset
    data = pd.read_csv('DataSet/south_asian_countries.csv')
    data = data.rename(columns={
        'Country': 'country',
        'Year': 'year',
        'GDP (current US$)': 'gdp',
        'GDP growth (annual %)': 'gdp_growth',
        'GDP per capita (current US$)': 'gdp_per_capita',
        'Unemployment, total (% of total labor force) (modeled ILO estimate)': 'unemployment',
        'Population, total': 'population',
        'Population growth (annual %)': 'population_growth',
        'Life expectancy at birth, total (years)': 'life_expectancy',
        'Mortality rate, infant (per 1,000 live births)': 'mortality'
    })
    # Remove unnecessary columns
    data = data.drop(columns=['Unnamed: 10', 'Unnamed: 11'], errors='ignore')
    return data

# Load data
data = load_data()

# Sidebar navigation
st.sidebar.header("Explore Analysis")
options = [
    "Population Trend",
    "GDP per Capita",
    "GDP",
    "Unemployment",
    "Life Expectancy",
    "Mortality Count",
    "GDP Prediction"
]
selected_option = st.sidebar.radio("Choose an analysis:", options)

# Group data into periods
def group_by_period(data, column):
    data['period'] = pd.cut(data['year'], bins=[1999, 2005, 2010, 2015, 2020], labels=['2000-2005', '2006-2010', '2011-2015', '2016-2020'])
    grouped = data.groupby(['country', 'period'])[column].mean().reset_index()
    return grouped

if selected_option == "Population Trend":
    st.header("🌍 Population Trend")
    grouped_data = group_by_period(data, 'population')
    fig = px.bar(grouped_data, x='period', y='population', color='country', barmode='group', title="Population Trend by Period")
    st.plotly_chart(fig)

elif selected_option == "GDP per Capita":
    st.header("💰 GDP per Capita")
    grouped_data = group_by_period(data, 'gdp_per_capita')
    fig = px.bar(grouped_data, x='period', y='gdp_per_capita', color='country', barmode='group', title="GDP per Capita by Period")
    st.plotly_chart(fig)

elif selected_option == "GDP":
    st.header("📈 GDP")
    grouped_data = group_by_period(data, 'gdp')
    fig = px.bar(grouped_data, x='period', y='gdp', color='country', barmode='group', title="GDP by Period")
    st.plotly_chart(fig)

elif selected_option == "Unemployment":
    st.header("📉 Unemployment")
    grouped_data = group_by_period(data, 'unemployment')
    fig = px.bar(grouped_data, x='period', y='unemployment', color='country', barmode='group', title="Unemployment by Period")
    st.plotly_chart(fig)

elif selected_option == "Life Expectancy":
    st.header("⏳ Life Expectancy")
    grouped_data = group_by_period(data, 'life_expectancy')
    fig = px.line(grouped_data, x='period', y='life_expectancy', color='country', markers=True, title="Life Expectancy by Period")
    st.plotly_chart(fig)

elif selected_option == "Mortality Count":
    st.header("⚰️ Mortality Count")
    grouped_data = group_by_period(data, 'mortality')
    fig = px.line(grouped_data, x='period', y='mortality', color='country', markers=True, title="Infant Mortality by Period")
    st.plotly_chart(fig)

elif selected_option == "GDP Prediction":
    st.header("🔮 GDP Prediction")
    
    # Feature selection
    features = ['population', 'gdp_growth', 'unemployment', 'life_expectancy', 'mortality']
    X = data[features].dropna()
    y = data['gdp'].dropna()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    st.write("Model Performance:")
    st.write(f"Mean Squared Error: {mse:.2f}")

    # Input for prediction
    st.subheader("Make a Prediction")
    population = st.number_input("Population", min_value=0, value=1000000, step=100000)
    gdp_growth = st.slider("GDP Growth (%)", -10.0, 10.0, 2.0)
    unemployment = st.slider("Unemployment (%)", 0.0, 50.0, 5.0)
    life_expectancy = st.slider("Life Expectancy (years)", 0.0, 100.0, 70.0)
    mortality = st.slider("Mortality Rate (per 1,000)", 0.0, 100.0, 10.0)

    input_data = [[population, gdp_growth, unemployment, life_expectancy, mortality]]
    predicted_gdp = model.predict(input_data)

    st.write(f"Predicted GDP: ${predicted_gdp[0]:,.2f}")
