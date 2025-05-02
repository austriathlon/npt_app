#!/Users/wadehobbs/Python_VS/wades_venv/bin/python
# coding: utf-8

## Set working directory
# cd /Users/wadehobbs/Python_VS
## Set the kernel to the virtual environment
# source wades_venv/bin/activate


import numpy as np
import snowflake.connector
from sqlalchemy import create_engine
import pandas as pd
import streamlit as st
import plotly.express as px



def clean_column_names(df):
    df.columns = (
        df.columns.str.strip()  # Remove leading/trailing whitespace
        .str.lower()  # Convert to lowercase
        .str.replace(' ', '_')  # Replace spaces with underscores
        .str.replace(r'[^\w\s]', '')  # Remove special characters
    )
    return df

st.logo(
    "AusTriathlon-Logo-500px.png",
    link="https://www.triathlon.org.au/",
    size="large"
)

# Set page config to use the full width of the screen
st.set_page_config(layout="wide")

pd.options.mode.copy_on_write = True

# Custom CSS to increase the font size of the selectbox label
st.markdown(
    """
    <style>
    /* Reduce the margin below the custom label */
    .ststSubheader {
        margin-bottom: -1rem;  /* Adjust this value as needed */
    }
    /* Adjust the spacing of the selectbox */
    div[data-baseweb="select"] {
        margin-top: -1.5rem;  /* Adjust this value as needed */
    }
    /* Adjust the sidebar width */
    .css-1d391kg {
        width: 180px;  /* Adjust the value as needed */
    }
    .title-container {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .title-text {
        flex-grow: 1;
        text-align: center;
    }
    .title-image {
        margin-right: 0px;  /* Adjust the spacing between the image and the title */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Create a layout with two columns
col1, col2 = st.columns([1, 8])

with col1:
    st.image("austriathlon_icon.png", width=100)

with col2:
    #st.title("AusTriathlon NPT App")
    st.markdown("<h1 style='text-align: center; '>AusTriathlon NPT Dashboard</h1>", unsafe_allow_html=True)

st.markdown("</br>", unsafe_allow_html=True)

# Connect to Snowflake
# Define connection parameters
# Access secrets
username = st.secrets["database"]["username"]
password = st.secrets["database"]["password"]
account = st.secrets["database"]["account"]
database = st.secrets["database"]["database"]
schema = st.secrets["database"]["schema"]
warehouse = st.secrets["database"]["warehouse"]
role = st.secrets["database"]["role"]

# Create a connection to the Snowflake database
engine = create_engine(
    f'snowflake://{username}:{password}@{account}/{database}/{schema}?warehouse={warehouse}&role={role}'
)

# Query the database and load the data into a DataFrame

query = """
SELECT program_id, event_id, program_date, program_name, race_level, 
       race_distance_calculated, event_title, athlete_id, athlete_title, 
       athlete_noc, rank, total_time, athlete_gender, splits, leg, secs_behind
FROM INDIVIDUAL_RESULT
WHERE program_date BETWEEN DATEADD(year, -1, CURRENT_DATE) AND CURRENT_DATE
"""
@st.cache_data
def load_data(query):
    return pd.read_sql(query, engine)

data = load_data(query)
#data = pd.read_sql(query, engine)

# Clean column names
data=clean_column_names(data)


# First mutate operation
# Data manipulation using pandas and numpy
data['race_level'] = np.where(data['event_title'].str.contains('olympic games', case=False, na=False), 'olympic games', data['race_level'])
data['race_level'] = np.where(data['event_title'].str.contains('olympic games test event', case=False, na=False), 'olympic qualification', data['race_level'])
data['race_level'] = np.where(data['event_title'].str.contains('commonwealth games', case=False, na=False), 'commonwealth games', data['race_level'])
data['race_level'] = np.where(data['race_level'].str.contains('major games', case=False, na=False), 'recognised games', data['race_level'])

# Filter rows based on 'program_name'
data = data[data['program_name'].isin(["elite women", "elite men", "u23 men", "u23 women", "junior men", "junior women"])]

# Filter out rows where 'event_title' contains specific strings
data = data[~data['event_title'].str.contains("duathlon", case=False, na=False)]
data = data[~data['event_title'].str.contains("winter", case=False, na=False)]
data = data[~data['event_title'].str.contains("aquathlon", case=False, na=False)]
data = data[~data['event_title'].str.contains("cross", case=False, na=False)]
data = data[~data['event_title'].str.contains("long distance", case=False, na=False)]

# Select specific columns
data2 = data[['program_id', 'event_id', 'program_date', 'program_name', 'race_level', 
             'race_distance_calculated', 'event_title', 'athlete_id', 'athlete_title', 
             'athlete_noc','rank', 'total_time', 'athlete_gender']]

# Remove duplicate rows
data2 = data2.drop_duplicates()


# Rank Data
from pandas import to_datetime
from pandas import DataFrame

# Query the RANKING_WORLD_MEN table and load the data into a DataFrame
query_ranking_world_men = """
SELECT *
FROM RANKING_WORLD_MEN
"""

@st.cache_data
def load_data(query):
    return pd.read_sql(query_ranking_world_men, engine)

ranking_world_men_data = load_data(query)

#ranking_world_men_data = pd.read_sql(query_ranking_world_men, engine)
ranking_world_men_data = clean_column_names(ranking_world_men_data)

ranking_world_men_data = (
    ranking_world_men_data
    .assign(
        athlete_title=lambda df: (df['athlete_first_name'] + ' ' + df['athlete_last_name']).str.lower(),
        date=lambda df: to_datetime(df['ranking_published_time'])
    )
    .drop(columns=['ranking_published_time'])
    .sort_values(by=['athlete_id', 'date'])
    .reset_index(drop=True)
    .rename(columns={'rank': 'world_rank'})
)

data2['program_date'] = pd.to_datetime(data2['program_date'])
data_male = data2[data2['athlete_gender'] == 'male']



ranking_world_men_data2 = ranking_world_men_data.drop(columns=['athlete_title', 'athlete_noc', 'athlete_gender'])

ranking_world_men_data2 = ranking_world_men_data2.rename(columns={
    'rank': 'world_rank',
    'athlete_id': 'ranking_athlete_id'
})


merged_data_male = pd.merge(
    data_male,
    ranking_world_men_data2,
    how='left',
    left_on='athlete_id',
    right_on='ranking_athlete_id'
)

# Filter rows based on the date range
merged_data_male = merged_data_male[
    (merged_data_male['program_date'] >= merged_data_male['dbt_valid_from']) &
    (merged_data_male['program_date'] <= merged_data_male['dbt_valid_to'])
]

# Clean column names
merged_data_male = clean_column_names(merged_data_male)

# Select specific columns
selected_columns = [
    'program_id', 'event_id', 'program_date', 'program_name', 'race_level', 
    'race_distance_calculated', 'event_title', 'athlete_id', 'athlete_title', 
    'rank', 'athlete_gender', 'ranking_category_name', 'ranking_name', 'world_rank', 
    'athlete_noc'
]

merged_data_male = merged_data_male[selected_columns]

# Drop duplicated columns
#merged_data_male = merged_data_male.loc[:, ~merged_data_male.columns.duplicated()]

# ## Womens Ranking


# Query the RANKING_WORLD_MEN table and load the data into a DataFrame
query_ranking_world_women = """
SELECT *
FROM RANKING_WORLD_WOMEN
"""

@st.cache_data
def load_data(query):
    return pd.read_sql(query_ranking_world_women, engine)

ranking_world_women_data = load_data(query)

#ranking_world_women_data = pd.read_sql(query_ranking_world_women, engine)
ranking_world_women_data = clean_column_names(ranking_world_women_data)

ranking_world_women_data = (
    ranking_world_women_data
    .assign(
        athlete_title=lambda df: (df['athlete_first_name'] + ' ' + df['athlete_last_name']).str.lower(),
        date=lambda df: to_datetime(df['ranking_published_time'])
    )
    .drop(columns=['ranking_published_time'])
    .sort_values(by=['athlete_id', 'date'])
    .reset_index(drop=True)
    .rename(columns={'rank': 'world_rank'})
)

data_female = data2[data2['athlete_gender'] == 'female']




# Merge data_male with ranking_world_men_data on athlete_id

ranking_world_women_data2 = ranking_world_women_data.drop(columns=['athlete_title', 'athlete_noc', 'athlete_gender'])

ranking_world_women_data2 = ranking_world_women_data2.rename(columns={
    'rank': 'world_rank',
    'athlete_id': 'ranking_athlete_id'
})


merged_data_female = pd.merge(
    data_female,
    ranking_world_women_data2,
    how='left',
    left_on='athlete_id',
    right_on='ranking_athlete_id'
)

# Filter rows based on the date range
merged_data_female = merged_data_female[
    (merged_data_female['program_date'] >= merged_data_female['dbt_valid_from']) &
    (merged_data_female['program_date'] <= merged_data_female['dbt_valid_to'])
]

# Clean column names
merged_data_female = clean_column_names(merged_data_female)

# Select specific columns
selected_columns = [
    'program_id', 'event_id', 'program_date', 'program_name', 'race_level', 
    'race_distance_calculated', 'event_title', 'athlete_id', 'athlete_title', 
    'rank', 'athlete_gender', 'ranking_category_name', 'ranking_name', 'world_rank', 
    'athlete_noc'
]

merged_data_female = merged_data_female[selected_columns]

# Drop duplicated columns
#merged_data_female = merged_data_female.loc[:, ~merged_data_female.columns.duplicated()]


# Function to calculate rank_cat, top_50_count, and qof_score
def calculate_scores(df):
    # Create rank_cat column
    df['rank_cat'] = np.where(df['world_rank'].between(1, 50), 1, 0)
    
    # Calculate top_50_count for each program_id
    df['top_50_count'] = df.groupby('program_id')['rank_cat'].transform('sum')
    
    # Calculate qof_score
    df['qof_score'] = df['top_50_count'] * 2
    
    return df

# Apply the function to both DataFrames
race_rank_male = calculate_scores(merged_data_male).reset_index(drop=True)
race_rank_female = calculate_scores(merged_data_female).reset_index(drop=True)

# Import the look up table to get NPT scores
npt_2024 = pd.read_csv("Draft2_NPT_2024.csv")

npt_2024 = clean_column_names(npt_2024)
npt_2024 = npt_2024.rename(columns={'class': 'race_class'})

# Select columns from 'class' to '80th'
selected_df = npt_2024.loc[:, 'race_class':'80th']

# Pivot longer
pivoted_df = selected_df.melt(id_vars=['race_class', 'qof'], var_name='name', value_name='value')

# Parse number from 'name' and create 'rank' column
pivoted_df['rank'] = pivoted_df['name'].str.extract(r'(\d+)').astype(int)

# Select specific columns
score_table = pivoted_df[['race_class', 'qof', 'rank', 'value']]


# Bind rows (concatenate DataFrames)
race_rank = pd.concat([race_rank_female, race_rank_male], ignore_index=True)

# Filter the DataFrame
race_rank_filtered = race_rank[
    ~race_rank['event_title'].str.contains("national") &
    (race_rank['race_distance_calculated'] != "super sprint") &
    (race_rank['race_level'] != "age-group event")
]

# Display the result
#print(race_rank_filtered)

# Define the conditions and choices for the case_when equivalent
conditions = [
    race_rank_filtered['race_level'].str.contains("olympic games", case=False, na=False),
    race_rank_filtered['race_level'].str.contains("grand final", case=False, na=False),
    race_rank_filtered['race_level'] == "world championship finals",
    race_rank_filtered['event_title'].str.contains("world triathlon championship series", case=False, na=False),
    race_rank_filtered['event_title'].str.contains("commonwealth games", case=False, na=False),
    race_rank_filtered['event_title'].str.contains("world triathlon cup", case=False, na=False),
    race_rank_filtered['event_title'].str.contains("u23 world championships", case=False, na=False),
    race_rank_filtered['event_title'].str.contains("oceania championships", case=False, na=False),
    race_rank_filtered['race_level'] == "regional championships",
    race_rank_filtered['race_level'].str.contains("continental|junior world championships", case=False, na=False),
    race_rank_filtered['race_level'].str.contains("continental", case=False, na=False)
]

choices = ["A", "A", "A", "B", "B", "C", "C", "C", "C", "D", "D"]

# Create the 'race_class' column using np.select
race_rank_filtered['race_class'] = np.select(conditions, choices, default=None)

# Perform the left join
race_rank_joined = pd.merge(race_rank_filtered, score_table, how='left', left_on=['race_class', 'rank', 'qof_score'], right_on=['race_class', 'rank', 'qof'])

# Select specific columns
selected_columns = [
    'event_title', 'program_date', 'program_name', 'race_level', 'race_distance_calculated', 
    'athlete_title', 'athlete_noc', 'athlete_gender', 'rank', 'world_rank', 'qof_score', 
    'race_class', 'value'
]
race_rank1 = race_rank_joined[selected_columns]

# This will remove duplicates of races ie devonport U23 and Elite
race_rank1 = (
    race_rank1.sort_values(by='value', ascending=False)  # Sort by 'value' in descending order
    .drop_duplicates(subset=['athlete_title', 'event_title'], keep='first')  # Keep the first occurrence for each group
)

# Group by athlete_title and calculate rank_value
race_rank1['rank_value'] = race_rank1.groupby('athlete_title')['value'].rank(method='first', ascending=False)

# Arrange by program_date
race_rank1['program_date'] = pd.to_datetime(race_rank1['program_date'])
race_rank1 = race_rank1.sort_values(by='program_date')

# Reset index
race_rank1 = race_rank1.reset_index(drop=True).sort_values(by="athlete_title")

# Sidebar for checkboxes and selectboxes
st.sidebar.header("Filter Options")

st.markdown("</br>", unsafe_allow_html=True)   

# Add title for the checkbox
# # Custom CSS to make sidebar markdown text unbolded
st.markdown(
    """
    <style>
    /* Make sidebar markdown text unbolded */
    .sidebar .markdown-text-container p {
        font-weight: normal !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
#st.sidebar.markdown("#### Nationality Filter")

# Streamlit checkboxes
show_australia = st.sidebar.checkbox('Show only Australians', value=True)

#Add title for radio buttons
# st.sidebar.subheader("Select Gender")

# Radio button for selecting gender
selected_gender = st.sidebar.radio('Select Gender', options=['All', 'Male', 'Female'], index=0, label_visibility='hidden')

# Add space between the radio buttons and the selectbox
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# st.markdown(
#     """<style>
# div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
#     font-size: 18px;
# }
#     </style>
#     """, unsafe_allow_html=True)


# Apply the "Australia" filter first
df_filtered = race_rank1.copy()

# Convert strings to Title case
string_columns = ['event_title', 'program_name', 'athlete_title', 'athlete_noc', 'race_class']
for col in string_columns:
    df_filtered[col] = df_filtered[col].fillna('').str.title()

# Set up the filters for the select boxes
if show_australia:
    df_filtered = df_filtered[df_filtered['athlete_noc'] == 'Aus']
if selected_gender != 'All':
    df_filtered = df_filtered[df_filtered['athlete_gender'].str.lower() == selected_gender.lower()]

# Extract unique, sorted names
filtered_names = df_filtered[['athlete_title']].drop_duplicates().sort_values(by="athlete_title")['athlete_title'].tolist()

# Set default to first name in the filtered list (if any exist)
default_index = 0 if filtered_names else None  # Avoid errors if list is empty

st.markdown("</br>", unsafe_allow_html=True)
selected_name = st.sidebar.selectbox('Select Athlete',filtered_names, index=default_index, label_visibility='visible')

# Select specific columns
selected_columns = [
    'event_title', 'program_date', 'program_name', 
    'athlete_title', 'athlete_noc', 'rank', 'world_rank', 'qof_score', 
    'race_class', 'value'
]
table_data = df_filtered[selected_columns]

# Replace None values with 0 in the 'value' column
table_data['value'] = table_data['value'].fillna(0)
table_data = table_data.sort_values(by='value', ascending=False)

columns_to_rename = {
    'event_title': 'Race',
    'program_date': 'Date',
    'program_name': 'Program',
    'athlete_title': 'Athlete',
    'athlete_noc': 'Country',
    'rank': 'Race Rank',
    'world_rank': 'World Rank',
    'qof_score': 'QOF Score',
    'race_class': 'Class',
    'value': 'NPT Score'
}
table_data = table_data.rename(columns=columns_to_rename)

# Format the 'Date' column to display only year-month-day
table_data['Date'] = pd.to_datetime(table_data['Date'], format='mixed').dt.strftime('%Y-%m-%d')

# Create tabs
tab1, tab2, tab3 = st.tabs(["NPT Individual Results", "NPT Top 100", "Race Results"])

with tab1:
    

    # Select box for choosing a name
    if filtered_names:  # Only show the selectbox if there are names available
        st.markdown("</br>", unsafe_allow_html=True)
        # Filter dataset based on selected name
        filtered_df = table_data[table_data['Athlete'] == selected_name]

        # Convert a specific column to upper case
        filtered_df['Country'] = filtered_df['Country'].str.upper()

        # Display the filtered athlete ratings to ensure the filtering is working correctly
        st.subheader("Results Table")
        
        # Display the filtered DataFrame
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        st.markdown("</br>", unsafe_allow_html=True)

        # Define custom colors for the levels
        color_discrete_map = {
        "A": "#F2B443",
        "B": "#007DFF",
        "C": "#803E80",
        "D": "#FF0000"
        }

        # Create the Plotly plot
        #plot_data = filtered_df[['NPT Score', 'Race Rank', 'Race', 'Class', 'QOF Score']].copy()

        st.subheader("Results Graph")
        fig = px.bar(
            filtered_df,
            x='NPT Score',
            y='Race',
            color='Class',
            text='QOF Score',
            #labels={'value': 'NPT Score', 'event_title': '', 'race_class': 'Class'},
            custom_data=['Race Rank', 'World Rank'],
            color_discrete_map=color_discrete_map,
            category_orders={"Class": ["A", "B", "C", "D"]}
        )
        fig.update_traces(
            hovertemplate='<b>Race Rank:</b> %{customdata[0]}<br><b>World Rank:</b> %{customdata[1]}<extra></extra>'
        )
        fig.update_layout(
            xaxis=dict(range=[0, 1250]),
            yaxis=dict(categoryorder='total ascending'),
            xaxis_title="NPT Score",
            yaxis_title="",
            legend_title="Class",
            bargap=0.2,
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            xaxis_gridwidth=0.5,
            yaxis_gridwidth=0.5,
            xaxis_gridcolor='gray',
            yaxis_gridcolor='gray'
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No names available based on the selected filters.")

with tab2:
    
    # Filter rows where rank_value is less than or equal to 3
    top3perathlete = race_rank1[race_rank1['rank_value'] <= 3]

    # Create table of summed top 3 NPT scores for all athletes
    athlete_ratings = (
    top3perathlete
    .groupby(['athlete_noc', 'athlete_title', 'athlete_gender'], as_index=False)
    .agg(score=('value', 'sum'))
    .assign(rank=lambda df: df.groupby('athlete_gender')['score'].rank(method='min', ascending=False))
    .loc[:, ['athlete_title', 'athlete_gender', 'athlete_noc', 'rank', 'score']]
    .sort_values(by='score', ascending=False)
    .reset_index(drop=True)

    )
    
    columns_to_rename = {
    'athlete_title': 'Athlete',
    'athlete_gender': 'Gender',
    'athlete_noc': 'Country',
    'rank': 'NPT Rank',
    'score': 'NPT Score'
    }
    athlete_ratings = athlete_ratings.rename(columns=columns_to_rename)

    # Filter the top 100 athletes' scores based on selected gender
    top_100 = athlete_ratings.copy()
    if show_australia:
        top_100 = top_100[top_100['Country'] == 'aus']
    if selected_gender != 'All':
        top_100 = top_100[top_100['Gender'].str.lower() == selected_gender.lower()]

    top_100_df = top_100.sort_values(by='NPT Score', ascending=False).head(100)

    string_columns2 = ['Athlete','Gender', 'Country']

    # Convert strings to Title case
    for col in string_columns2:
        top_100_df[col] = top_100_df[col].fillna('').str.title()

    # Convert a specific column to upper case
    top_100_df['Country'] = top_100_df['Country'].str.upper()

    st.subheader("Top 100 NPT Scores")
    st.dataframe(top_100_df, use_container_width=True, hide_index=True)

with tab3:
    # Select specific columns
    data3 = data[['program_id', 'event_id', 'program_date', 'program_name', 'race_level', 
             'race_distance_calculated', 'event_title', 'athlete_id', 'athlete_title', 'athlete_gender',
             'athlete_noc','rank', 'total_time', 'splits', 'leg', 'secs_behind']]
    df_filtered = data3.copy()


    # Convert strings to Title case
    string_columns = ['event_title', 'program_name', 'athlete_title', 'athlete_noc', 'athlete_gender']
    for col in string_columns:
        df_filtered[col] = df_filtered[col].fillna('').str.title()

    # Set up the filters for the select boxes
    if show_australia:
        df_filtered = df_filtered[df_filtered['athlete_noc'] == 'Aus']


    # Extract unique, sorted names
    #filtered_names = df_filtered[['athlete_title']].drop_duplicates().sort_values(by="athlete_title")['athlete_title'].tolist()

    # Set default to first name in the filtered list (if any exist)
    #default_index = 0 if filtered_names else None  # Avoid errors if list is empty

    # Select specific columns
    selected_columns = [
        'event_title', 'program_date', 'program_name', 
        'athlete_title', 'athlete_noc', 'rank', 'total_time',
        'leg', 'splits', 'secs_behind'
    ]
    table_data = df_filtered[selected_columns]

    # melt the table to be longer so splits and secs behind have their own columns
    table_data_wide = table_data.pivot_table(
        index=['event_title', 'program_date', 'program_name', 'athlete_title', 'athlete_noc', 'rank', 'total_time'],
        columns='leg',
        values=['splits', 'secs_behind']
    )

    # Flatten the MultiIndex columns
    table_data_wide.columns = ['_'.join(col).strip() for col in table_data_wide.columns.values]
    table_data_wide.reset_index(inplace=True)


    columns_to_rename = {
    'event_title': 'Race',
    'program_date': 'Date',
    'program_name': 'Program',
    'athlete_title': 'Athlete',
    'athlete_noc': 'Country',
    'rank': 'Rank',
    'total_time': 'Total Time',
    'secs_behind_swim': 'TBF Swim',
    'secs_behind_bike': 'TBF Bike',
    'secs_behind_run': 'TBF Run',
    'splits_swim': 'Swim',
    'splits_bike': 'Bike',
    'splits_run': 'Run',
    }
    table_data_wide2 = table_data_wide.rename(columns=columns_to_rename)

    # Reordering columns using loc
    table_data_wide2 = table_data_wide2.loc[:, ['Race', 'Date', 'Program', 'Athlete', 'Country', 'Rank', 'Total Time', 'Swim','TBF Swim', 'Bike', 'TBF Bike', 'Run','TBF Run']]
    # Convert 'Total Time' and all columns between 'Total Time' and 'TBF Run' to datetime format and then to string format
    columns_to_convert = table_data_wide2.loc[:, 'Total Time':'TBF Run'].columns

    table_data_wide2[columns_to_convert] = table_data_wide2[columns_to_convert].apply(
        lambda x: pd.to_datetime(x, unit='s').dt.strftime('%H:%M:%S')
    )

    display_columns_table2 = ['Race', 'Date', 'Program', 'Rank', 'Total Time', 'Swim','TBF Swim', 'Bike', 'TBF Bike', 'Run','TBF Run']

        # Select box for choosing a name
    if filtered_names:  # Only show the selectbox if there are names available
        #selected_name = st.sidebar.selectbox('test',filtered_names, index=default_index, label_visibility='hidden')
        # Filter dataset based on selected name
        filtered_df = table_data_wide2.sort_values(['Date', 'Program', 'Rank'], ascending=[False, True, True])
        filtered_df2 = filtered_df[filtered_df['Athlete'] == selected_name]
        
        # Display the filtered DataFrame
        st.dataframe(filtered_df2[display_columns_table2], use_container_width=True, hide_index=True)
        #st.write(filtered_df2.to_html(index=False), unsafe_allow_html=True)
        # Display the filtered DataFrame using AgGrid
        # gb = GridOptionsBuilder.from_dataframe(filtered_df2)
        # gb.configure_pagination(paginationAutoPageSize=False)  # Enable pagination
        # gb.configure_side_bar()  # Enable a sidebar
        # gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
        # gridOptions = gb.build()

        # AgGrid(filtered_df2, gridOptions=gridOptions, enable_enterprise_modules=True)
