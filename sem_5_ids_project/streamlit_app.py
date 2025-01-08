import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="South Asian Countries Analysis",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("South Asian Countries economic analysis")
st.write("In this app we analyze different statistics of south asian country that can help grow thier economy")

# Sidebar
st.sidebar.header("Navigation")
menu_options = ["Home🏠",'Population Analysis🌍', "GDP growth📈", "Mortality Rate⚰️","Unemployment💼", "Life Expectancy rate🌱","predict GDP of a country🤖"]
selected_option = st.sidebar.radio("Choose a section:", menu_options)

# Function to load uploaded data
@st.cache_data
def load_uploaded_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

data = load_uploaded_data("archive/South_Asian_dataset.csv")

# Main sections
## Home Page
# Home Page

# Home Page
if selected_option == "Home🏠":
    st.subheader("Welcome to the Home Page!")
    st.write("Below is a list of unique South Asian countries and their current GDP for the year 2023.")

    # Check if data is loaded
    if data is not None:
        # Filter data for the year 2023
        filtered_data = data[data['Year'] == 2023]

        # Convert 'GDP (current US$)' to numeric, handling invalid entries
        filtered_data['GDP (current US$)'] = pd.to_numeric(filtered_data['GDP (current US$)'], errors='coerce')

        # Replace NaN values in GDP with 0
        filtered_data['GDP (current US$)'].fillna(0, inplace=True)

        # Replace 0 GDP values with the average GDP of other countries
        non_zero_gdp = filtered_data[filtered_data['GDP (current US$)'] > 0]['GDP (current US$)']
        average_gdp = non_zero_gdp.mean()
        filtered_data.loc[filtered_data['GDP (current US$)'] == 0, 'GDP (current US$)'] = average_gdp

        # Debug: Display the first few rows of the filtered dataset
        st.write("### Debug: Filtered Dataset for 2023 (Zero GDP Replaced)")
        st.dataframe(filtered_data.head(), use_container_width=True)

        # Extract unique countries
        unique_countries = filtered_data['Country'].drop_duplicates().sort_values()
        st.write("### Unique Countries")
        st.write(", ".join(unique_countries))  # Display countries as a comma-separated list

        # Display countries and their current GDP for 2023
        st.write("### Countries and their Current GDP (2023)")
        current_gdp_df = filtered_data[['Country', 'GDP (current US$)']].drop_duplicates().reset_index(drop=True)
        st.dataframe(current_gdp_df, use_container_width=True)

        # Show average GDP used for replacement
        st.write(f"### Average GDP used for replacement: {average_gdp:,.2f}")
    else:
        st.warning("Data not loaded. Please upload the dataset.")


# GDP Growth Page
if selected_option == "GDP growth📈":
    st.subheader("GDP Growth Analysis")
    st.write("Below is the average GDP of South Asian countries for different time intervals.")

    # Check if data is loaded
    if data is not None:
        # Ensure 'Year' and 'GDP (current US$)' columns are numeric
        data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
        data['GDP (current US$)'] = pd.to_numeric(data['GDP (current US$)'], errors='coerce')

        # Define intervals
        intervals = [
            (2000, 2005),
            (2006, 2010),
            (2011, 2015),
            (2016, 2020),
            (2021, 2023)
        ]

        # Prepare data for intervals
        gdp_growth_data = []
        for start, end in intervals:
            interval_data = data[(data['Year'] >= start) & (data['Year'] <= end)]
            interval_avg_gdp = interval_data.groupby('Country')['GDP (current US$)'].mean().reset_index()
            interval_avg_gdp['Interval'] = f"{start}-{end}"
            gdp_growth_data.append(interval_avg_gdp)

        # Concatenate all interval data
        gdp_growth_df = pd.concat(gdp_growth_data, ignore_index=True)

        # Replace 0 values with the average GDP for each country
        country_avg_gdp = gdp_growth_df.groupby('Country')['GDP (current US$)'].mean()
        gdp_growth_df['GDP (current US$)'] = gdp_growth_df.apply(
            lambda row: country_avg_gdp[row['Country']] if row['GDP (current US$)'] == 0 else row['GDP (current US$)'],
            axis=1
        )

        # Pivot for better display
        pivot_table = gdp_growth_df.pivot(index='Country', columns='Interval', values='GDP (current US$)')
        pivot_table.fillna(0, inplace=True)  # Replace NaN values with 0 for better readability

        # Display the GDP growth data
        st.write("### Average GDP for Each Interval (0 Replaced with Country Average)")
        st.dataframe(pivot_table, use_container_width=True)

        # Visualization using Streamlit's chart
        st.write("### GDP Growth Visualization")
        st.bar_chart(pivot_table)

    else:
        st.warning("Data not loaded. Please upload the dataset.")


# Population Analysis Page
# Population Analysis Page
# Streamlit code for Population Analysis Page
elif selected_option == "Population Analysis🌍":
    st.subheader("Population Analysis")
    st.write("Below is the average population of South Asian countries for different time intervals.")

    # Check if data is loaded
    if data is not None:
        # Ensure 'Year' and 'Population, total' columns are numeric
        data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
        data['Population, total'] = pd.to_numeric(data['Population, total'], errors='coerce')

        # Remove rows with missing or zero population data
        data['Population, total'].replace(0, None, inplace=True)

        # Create 5-year intervals
        bins = [2000, 2005, 2010, 2015, 2020, 2023]
        labels = ['2000-2005', '2006-2010', '2011-2015', '2016-2020', '2021-2023']
        data['Interval'] = pd.cut(data['Year'], bins=bins, labels=labels, right=True)

        # Calculate average population for each interval and country
        avg_population = data.groupby(['Country', 'Interval'])['Population, total'].mean().reset_index()

        # Pivot for better display
        pivot_data = avg_population.pivot(index='Interval', columns='Country', values='Population, total')

        # Replace NaN with 0 for cleaner display
        pivot_data.fillna(0, inplace=True)

        # Display data as a table
        st.write("### Average Population for Each Interval")
        st.dataframe(pivot_data, use_container_width=True)

        # Create a bar chart
        st.write("### Population Trends")
        pivot_data.plot(kind='bar', figsize=(12, 8), width=0.8)
        plt.title('Population (2000-2023)', fontsize=14)
        plt.xlabel('Interval', fontsize=12)
        plt.ylabel('Average Population (in millions)', fontsize=12)
        plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        plt.tight_layout()

        # Show the plot in Streamlit
        st.pyplot(plt)

    else:
        st.warning("Data not loaded. Please upload the dataset.")

# Streamlit code for Mortality Rate Analysis Page
elif selected_option == "Mortality Rate⚰️":
    st.subheader("Infant Mortality Rate Analysis")
    st.write("Below is the average infant mortality rate (per 1,000 live births) for South Asian countries over different time intervals.")

    # Check if data is loaded
    if data is not None:
        # Ensure 'Year' and mortality rate column are numeric
        data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
        data['Mortality rate, infant (per 1,000 live births)'] = pd.to_numeric(data['Mortality rate, infant (per 1,000 live births)'], errors='coerce')

        # Remove rows with missing or zero mortality rate data
        data['Mortality rate, infant (per 1,000 live births)'].replace(0, None, inplace=True)

        # Create 5-year intervals
        bins = [2000, 2005, 2010, 2015, 2020, 2023]
        labels = ['2000-2005', '2006-2010', '2011-2015', '2016-2020', '2021-2023']
        data['Interval'] = pd.cut(data['Year'], bins=bins, labels=labels, right=True)

        # Handle outliers for Pakistan
        outlier_threshold = 100
        data.loc[
            (data['Country'] == 'Pakistan') &
            (data['Mortality rate, infant (per 1,000 live births)'] > outlier_threshold),
            'Mortality rate, infant (per 1,000 live births)'
        ] = None

        # Calculate average mortality rates for each interval and country
        avg_mortality = data.groupby(['Country', 'Interval'])['Mortality rate, infant (per 1,000 live births)'].mean().reset_index()

        # Pivot data for display
        pivot_data = avg_mortality.pivot(index='Interval', columns='Country', values='Mortality rate, infant (per 1,000 live births)')

        # Replace NaN with 0 for cleaner display
        pivot_data.fillna(0, inplace=True)

        # Display data as a table
        st.write("### Average Infant Mortality Rate for Each Interval")
        st.dataframe(pivot_data, use_container_width=True)

        # Create a bar chart
        st.write("### Infant Mortality Trends")
        pivot_data.plot(kind='bar', figsize=(12, 8), width=0.8)
        plt.title('Average Infant Mortality Rate (2000-2023)', fontsize=14)
        plt.xlabel('Interval', fontsize=12)
        plt.ylabel('Average Infant Mortality Rate (per 1,000 live births)', fontsize=12)
        plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        plt.tight_layout()

        # Show the plot in Streamlit
        st.pyplot(plt)
    else:
        st.warning("Data not loaded. Please upload the dataset.")

elif selected_option == "Unemployment💼":
    st.subheader("Unemployment Analysis")
    
    if data is not None:
        # Correct column name for unemployment
        unemployment_column = 'Unemployment, total (% of total labor force) (modeled ILO estimate)'
        
        # Create 5-year intervals
        data['Interval'] = pd.cut(
            data['Year'],
            bins=[2000, 2005, 2010, 2015, 2020, 2023],
            labels=['2000-2005', '2006-2010', '2011-2015', '2016-2020', '2021-2023'],
            right=True
        )
        
        # Threshold for outlier exclusion
        outlier_threshold = 20
        
        # Replace outliers in Pakistan's unemployment data with NaN
        data.loc[
            (data['Country'] == 'Pakistan') & 
            (data[unemployment_column] > outlier_threshold), 
            unemployment_column
        ] = None
        
        # Recalculate average unemployment rates for each 5-year interval
        avg_unemployment = data.groupby(['Country', 'Interval'])[unemployment_column].mean().reset_index()
        
        # Pivot for better display
        pivot_table = avg_unemployment.pivot(index='Country', columns='Interval', values=unemployment_column)
        pivot_table.fillna(0, inplace=True)  # Replace NaN values with 0 for now
        
        # Replace zeros in the Pakistan row with the average of the respective column
        pakistan_row = pivot_table.loc['Pakistan']
        for col in pivot_table.columns:
            if pakistan_row[col] == 0:
                # Calculate the average of the column excluding zeros
                col_avg = pivot_table[col].replace(0, pd.NA).mean(skipna=True)
                pivot_table.loc['Pakistan', col] = col_avg

        # Display the pivot table
        st.write("Unemployment Rates Pivot Table (Adjusted)")
        st.dataframe(pivot_table)

        # Visualize the pivot table
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_table.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Average Unemployment Rate by Interval and Country (2000-2023)', fontsize=14)
        ax.set_xlabel('Country', fontsize=12)
        ax.set_ylabel('Average Unemployment Rate (%)', fontsize=12)
        ax.legend(title='Interval', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        
    else:
        st.warning("Please upload a dataset to visualize unemployment data.")


elif selected_option == "Life Expectancy rate🌱":
    st.subheader("Life Expectancy Analysis")
    
    if data is not None:
        # Correct column name for life expectancy
        life_expectancy_column = 'Life expectancy at birth, total (years)'
        
        # Create 5-year intervals
        data['Interval'] = pd.cut(
            data['Year'],
            bins=[2000, 2005, 2010, 2015, 2020, 2023],
            labels=['2000-2005', '2006-2010', '2011-2015', '2016-2020', '2021-2023'],
            right=True
        )
        
        # Calculate average life expectancy for each 5-year interval
        avg_life_expectancy = data.groupby(['Country', 'Interval'])[life_expectancy_column].mean().reset_index()
        
        # Pivot for better display
        pivot_table = avg_life_expectancy.pivot(index='Country', columns='Interval', values=life_expectancy_column)
        pivot_table.fillna(0, inplace=True)  # Replace NaN values with 0 for now
        
        # Replace zeros in the Pakistan row with the average of the respective column
        pakistan_row = pivot_table.loc['Pakistan']
        for col in pivot_table.columns:
            if pakistan_row[col] == 0:
                # Calculate the average of the column excluding zeros
                col_avg = pivot_table[col].replace(0, pd.NA).mean(skipna=True)
                pivot_table.loc['Pakistan', col] = col_avg

        # Display the pivot table
        st.write("Life Expectancy Rates Pivot Table (Adjusted)")
        st.dataframe(pivot_table)

        # Visualize the pivot table
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_table.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Average Life Expectancy by Interval and Country (2000-2023)', fontsize=14)
        ax.set_xlabel('Country', fontsize=12)
        ax.set_ylabel('Average Life Expectancy (Years)', fontsize=12)
        ax.legend(title='Interval', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        
    else:
        st.warning("Please upload a dataset to visualize life expectancy data.")


elif selected_option == "predict GDP of a country🤖":
    st.subheader("Predict GDP of a Country")
    
    # Prompt user for inputs
    st.write("Provide the following inputs to predict the GDP:")
    country_name = st.text_input("Country Name")
    population = st.number_input("Total Population", min_value=0, step=1, format="%d")
    mortality_rate = st.number_input("Mortality Rate (infant, per 1,000 live births)", min_value=0.0, step=0.1)
    life_expectancy = st.number_input("Life Expectancy at Birth (years)", min_value=0.0, step=0.1)
    
    # Button to predict GDP
    if st.button("Predict GDP"):
        if population > 0 and mortality_rate > 0 and life_expectancy > 0:
            try:
                # Load the dataset
                df = pd.read_csv("archive/South_Asian_dataset.csv")

                # Select relevant columns
                df = df[['GDP (current US$)', 'Population, total', 
                         'Mortality rate, infant (per 1,000 live births)', 
                         'Life expectancy at birth, total (years)']]

                # Drop missing values
                df = df.dropna()

                # Clean the GDP column
                df['GDP (current US$)'] = (
                    df['GDP (current US$)']
                    .replace(',', '', regex=True)  # Remove commas
                    .astype(str)                   # Ensure it's a string for further processing
                    .str.replace('E', 'e', regex=False)  # Normalize scientific notation
                    .astype(float)                 # Convert to float
                )

                # Define features and target
                features = ['Population, total', 
                            'Mortality rate, infant (per 1,000 live births)', 
                            'Life expectancy at birth, total (years)']
                target = 'GDP (current US$)'

                X = df[features]
                y = df[target]

                # Train the model
                model = RandomForestRegressor(random_state=42)
                model.fit(X, y)

                # Predict GDP for user inputs
                new_data = pd.DataFrame({
                    'Population, total': [population],
                    'Mortality rate, infant (per 1,000 live births)': [mortality_rate],
                    'Life expectancy at birth, total (years)': [life_expectancy]
                })

                predicted_gdp = model.predict(new_data)
                st.success(f"The predicted GDP for {country_name} is: ${predicted_gdp[0]:,.2f}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Please provide valid inputs for all fields.")


# Footer
st.markdown(
    """
    <hr style='border: 1px solid #e0e0e0;'>
    <footer style='text-align: center; font-size: small; color: #777;'>
    </footer>
    """,
    unsafe_allow_html=True,
)
