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
    rank, athlete_gender, ranking_category_name, ranking_name, world_rank, 
    athlete_noc,  athlete_yob, duathlon, qof_score, race_class, npt_score, npt_type
FROM INDIVIDUAL_RESULT_WITH_NPT
WHERE program_date BETWEEN DATEADD(year, -5, CURRENT_DATE) AND CURRENT_DATE
"""
@st.cache_data(ttl=86400)
def load_data(query):
    return pd.read_sql(query, engine)

data = load_data(query)
#data = pd.read_sql(query, engine)

# Clean column names
data=clean_column_names(data)


# Filter the DataFrame
data = data[
    ~data['event_title'].str.contains("national", case=False, na=False) &
    (data['race_distance_calculated'] != "super sprint") &
    (data['race_level'] != "age-group event")
]



# Filter rows based on 'program_name'
data = data[data['program_name'].isin(["elite women", "elite men", "u23 men", "u23 women", "junior men", "junior women"])]

# Filter out rows where 'event_title' contains specific strings
data = data[~data['event_title'].str.contains("duathlon", case=False, na=False)]
data = data[~data['event_title'].str.contains("winter", case=False, na=False)]
data = data[~data['event_title'].str.contains("aquathlon", case=False, na=False)]
data = data[~data['event_title'].str.contains("cross", case=False, na=False)]
data = data[~data['event_title'].str.contains("long distance", case=False, na=False)]

#title for filter options in side bar
st.sidebar.header("Filter Options")
st.sidebar.markdown("<br>", unsafe_allow_html=True)

#Date selector for the sidebar
# After loading data
# Date selector for the sidebar
import datetime

# Ensure program_date is datetime
data['program_date'] = pd.to_datetime(data['program_date'])

# Set default end date to today or latest in data
# --- Place this at the very top of your sidebar filter section ---

# Set default end date to today or latest in data
#default_end = min(pd.Timestamp.today().date(), data['program_date'].max().date())
default_end = pd.Timestamp.today().date()
future_limit = pd.Timestamp.today().date() + pd.DateOffset(years=1)

# Use session state to store the selected end date
if 'selected_end' not in st.session_state:
    st.session_state.selected_end = default_end


st.sidebar.markdown(
    "<div style='font-weight:600; font-size:1.0em; margin-bottom:-10rem;'>End Date (1-year window)</div>",
    unsafe_allow_html=True
)

# Sidebar date input for end date
selected_end = st.sidebar.date_input(
    "",
    value=st.session_state.selected_end,
    min_value=data['program_date'].min().date(),
    max_value=future_limit
)

# Button to reset to today (default_end)
if st.sidebar.button("Reset to Today"):
    st.session_state.selected_end = default_end
    st.rerun()
else:
    # Only update session state if the user picked a new date
    if selected_end != st.session_state.selected_end:
        st.session_state.selected_end = selected_end
        st.rerun()

# Calculate start date based on selected end date (always a 1-year window)
selected_start = pd.Timestamp(st.session_state.selected_end) - pd.DateOffset(years=1) + pd.DateOffset(days=1)
selected_end_ts = pd.Timestamp(st.session_state.selected_end)

# Show the window to the user
#st.sidebar.markdown(f"**Window:** {selected_start.date()} to {selected_end_ts.date()}")

# Now filter your data using Timestamps for correct comparison
data = data[
    (data['program_date'] >= selected_start) &
    (data['program_date'] <= selected_end_ts)
]


#age selector for the sidebar
current_year = pd.Timestamp.now().year
data['athlete_age'] = current_year - data['athlete_yob']

def assign_age_group(age):
    if age <= 18:
        return '18'
    elif 18 < age <= 21:
        return '21'
    elif 21 < age <= 23:
        return '23'
    elif 23 < age <= 25:
        return '25'
    else:
        return 'Senior'

data['age_group'] = data['athlete_age'].apply(assign_age_group)


# Add space between the radio buttons and the selectbox
st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.markdown("#### Athlete Age")
age_group_options = ['All', '18', '21', '23', '25', 'Senior']
selected_age_group = st.sidebar.selectbox('Select Age Group', age_group_options, index=0, label_visibility="visible")
# Add space between the radio buttons and the selectbox
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Filter data based on selected age group
if selected_age_group != 'All':
    data = data[data['age_group'] == selected_age_group]

# Add radio button for duathlon filter
# duathlon_filter = st.sidebar.radio(
#     "Exclude Duathlon Races?",
#     options=["Include All Races", "Exclude Duathlon"],
#     index=1,
#     label_visibility='hidden'  # Default to "Exclude Duathlon"
# )
# if duathlon_filter == "Exclude Duathlon":
#     data = data[data['npt_type'] == 'no_duathlon']
# else:
#     data = data[data['npt_type'] == 'all_races']

# Add checkbox for duathlon filter
exclude_duathlon = st.sidebar.checkbox("Exclude Duathlon Races", value=True)

if exclude_duathlon:
    data = data[data['npt_type'] == 'no_duathlon']
else:
    data = data[data['npt_type'] == 'all_races']


# Add space between the radio buttons and the selectbox
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Remove duplicate rows
race_rank1 = data.drop_duplicates()


# This will remove duplicates of races ie devonport U23 and Elite
race_rank1 = (
    race_rank1.sort_values(by='npt_score', ascending=False)  # Sort by 'value' in descending order
    .drop_duplicates(subset=['athlete_title', 'event_title'], keep='first')  # Keep the first occurrence for each group
)

# Group by athlete_title and calculate rank_value
race_rank1['rank_value'] = race_rank1.groupby('athlete_title')['npt_score'].rank(method='first', ascending=False)

# Arrange by program_date
race_rank1['program_date'] = pd.to_datetime(race_rank1['program_date'])
race_rank1 = race_rank1.sort_values(by='program_date')

# Reset index
race_rank1 = race_rank1.reset_index(drop=True).sort_values(by="athlete_title")

# Sidebar for checkboxes and selectboxes

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


# Apply the "Australia" filter first
df_filtered = race_rank1.copy()

# Convert strings to Title case
string_columns = ['event_title', 'race_distance_calculated', 'program_name', 'athlete_title', 'athlete_noc', 'race_class']
for col in string_columns:
    df_filtered[col] = df_filtered[col].fillna('').str.title()

# Set up the filters for the select boxes
if show_australia:
    df_filtered = df_filtered[df_filtered['athlete_noc'] == 'Aus']
if selected_gender != 'All':
    df_filtered = df_filtered[df_filtered['athlete_gender'].str.lower() == selected_gender.lower()]

# Extract unique, sorted names
# Build the filtered_names list as you do now
filtered_names = df_filtered[['athlete_title']].drop_duplicates().sort_values(by="athlete_title")['athlete_title'].tolist()

# Initialize session state for athlete selection if not present
if 'selected_athlete' not in st.session_state:
    st.session_state.selected_athlete = None

# If the current selected athlete is not in the filtered list, reset to None
if st.session_state.selected_athlete not in filtered_names:
    st.session_state.selected_athlete = None

# Show selectbox, using session state as value
selected_name = st.sidebar.selectbox(
    'Select Athlete',
    filtered_names,
    index=filtered_names.index(st.session_state.selected_athlete) if st.session_state.selected_athlete in filtered_names else None,
    placeholder="To begin, please select an athlete",
    key="athlete_selectbox"
)

# Only update session state and rerun if the user picked a new athlete
if selected_name != st.session_state.selected_athlete:
    st.session_state.selected_athlete = selected_name
    st.rerun()
    
# Set default to first name in the filtered list (if any exist)
#default_index = 0 if filtered_names else None  # Avoid errors if list is empty

st.markdown("</br>", unsafe_allow_html=True)
#selected_name = st.sidebar.selectbox('Select Athlete',filtered_names, index=default_index, label_visibility='visible')

# Set default to None (no athlete selected initially)
# selected_name = st.sidebar.selectbox(
#     'Select Athlete',
#     filtered_names,
#     index=None,
#     placeholder="To begin, please select an athlete"
# )

st.markdown("</br>", unsafe_allow_html=True)

# Select specific columns
selected_columns = [
    'event_title', 'race_distance_calculated', 'program_date', 'program_name', 
    'athlete_title', 'athlete_noc', 'athlete_yob', 'rank', 'world_rank', 'qof_score', 
    'race_class', 'npt_score'
]
table_data = df_filtered[selected_columns]

# Replace None values with 0 in the 'value' column
table_data['npt_score'] = table_data['npt_score'].fillna(0)
table_data = table_data.sort_values(by='npt_score', ascending=False)

columns_to_rename = {
    'event_title': 'Race',
    'race_distance_calculated': 'Distance',
    'program_date': 'Date',
    'program_name': 'Program',
    'athlete_title': 'Athlete',
    'athlete_noc': 'Country',
    'athlete_yob': 'Year of Birth',
    'rank': 'Race Rank',
    'world_rank': 'World Rank',
    'qof_score': 'QOF Score',
    'race_class': 'Class',
    'npt_score': 'NPT Score'
}
table_data = table_data.rename(columns=columns_to_rename)

# Format the 'Date' column to display only year-month-day
table_data['Date'] = pd.to_datetime(table_data['Date'], format='mixed').dt.strftime('%Y-%m-%d')

# get age-group specific NPT scores, function is used in tab 1 and 2 to get filtered NPT scores
def get_highlight_indices(df):
    """Return indices of rows that would be highlighted for a given athlete DataFrame."""
    if df.empty:
        return []
    current_year = datetime.datetime.now().year
    yob = df['Year of Birth'].iloc[0]
    highlight_idx = []
    if yob <= current_year - 23:
        std = df[df['Distance'].str.lower() == 'standard']
        top2_std_idx = std.nlargest(2, 'NPT Score').index.tolist()
        rest = df.drop(top2_std_idx)
        top_other_idx = rest['NPT Score'].idxmax() if not rest.empty else None
        highlight_idx = top2_std_idx
        if top_other_idx is not None:
            highlight_idx.append(top_other_idx)
    elif (current_year - 23) < yob <= (current_year - 21):
        std = df[df['Distance'].str.lower() == 'standard']
        top_std_idx = std['NPT Score'].idxmax() if not std.empty else None
        rest = df.drop([top_std_idx]) if top_std_idx is not None else df
        top2_other_idx = rest.nlargest(2, 'NPT Score').index.tolist()
        highlight_idx = []
        if top_std_idx is not None:
            highlight_idx.append(top_std_idx)
        highlight_idx.extend(top2_other_idx)
    elif yob >= current_year - 20:
        highlight_idx = df.nlargest(3, 'NPT Score').index.tolist()
    return highlight_idx


# Create tabs
tab1, tab2 = st.tabs(["NPT Individual Results", "NPT Top 100"])

import datetime

with tab1:
    if not selected_name:
        st.info("To begin, please select an athlete from the sidebar.")
    else:
        # Select box for choosing a name
        if filtered_names:  # Only show the selectbox if there are names available
            st.markdown("</br>", unsafe_allow_html=True)
            # Filter dataset based on selected name
            filtered_df = table_data[table_data['Athlete'] == selected_name]

            # Convert a specific column to upper case
            filtered_df['Country'] = filtered_df['Country'].str.upper()
    
            # Display the filtered athlete ratings to ensure the filtering is working correctly
            st.subheader("Results Table")
            
            # --- Highlighting function ---
            def highlight_top_races(df):
                highlight_idx = get_highlight_indices(df)
                def highlight_row(row):
                    if row.name in highlight_idx:
                        return ['background-color: #d6f5d6'] * len(row)
                    else:
                        return [''] * len(row)
                numeric_cols = df.select_dtypes(include=['number']).columns
                styled = df.style.apply(highlight_row, axis=1).format({col: "{:.0f}" for col in numeric_cols})
                return styled

            # --- Use the function and display ---
            styled_df = highlight_top_races(filtered_df)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            st.markdown("</br>", unsafe_allow_html=True)
            # After displaying the styled table
            highlight_idx = get_highlight_indices(filtered_df)
            highlighted_sum = filtered_df.loc[highlight_idx, 'NPT Score'].sum() if highlight_idx else 0

            st.markdown(
                f"<div style='text-align:right; font-size:1em; margin-top:-15px;'>"
                f"<b>Adjusted NPT Score Total:</b> {highlighted_sum:.0f}"
                f"</div>",
                unsafe_allow_html=True
            )
            
            # Add extra space before the graph
            st.markdown("<br>", unsafe_allow_html=True)


    
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
        .agg(score=('npt_score', 'sum'))
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
    athlete_ratings['Athlete'] = athlete_ratings['Athlete'].str.strip().str.title()
    table_data['Athlete'] = table_data['Athlete'].str.strip().str.title()
    # --- Add column for sum of green-highlighted rows (Adjusted NPT Score) ---
    def get_top_npt_sum(table_data):
        results = {}
        for athlete, group in table_data.groupby('Athlete'):
            idx = get_highlight_indices(group)
            results[athlete] = group.loc[idx, 'NPT Score'].sum()
        return results

    # Use the table_data DataFrame from earlier (with renamed columns)
    top_npt_sum_dict = get_top_npt_sum(table_data)
    
    athlete_ratings['Adjusted NPT Score'] = athlete_ratings['Athlete'].map(top_npt_sum_dict)
      
    # Filter the top 100 athletes' scores based on selected gender and country
    top_100 = athlete_ratings.copy()
    if show_australia:
        top_100 = top_100[top_100['Country'] == 'aus']
    if selected_gender != 'All':
        top_100 = top_100[top_100['Gender'].str.lower() == selected_gender.lower()]

    top_100_df = top_100.sort_values(by='NPT Score', ascending=False).head(100)

    string_columns2 = ['Athlete', 'Gender', 'Country']
    for col in string_columns2:
        top_100_df[col] = top_100_df[col].fillna('').str.title()

    top_100_df['Country'] = top_100_df['Country'].str.upper()

    st.subheader("Top 100 NPT Scores")
    st.dataframe(top_100_df, use_container_width=True, hide_index=True)

    st.markdown(
    """
    <br>
    <span style="font-size: 1.1em;">
    The <b>'Adjusted NPT Score'</b> column includes a minimum number of standard distance races for each age category as follows:<br>
    &nbsp;&nbsp;&nbsp;&nbsp;- <b>Senior</b>: minimum 2 standard distance races to count towards total score.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;- <b>U23</b>: minimum 1 standard distance race to count towards total score; exempt 1st year U23 athlete who are transitioning from junior category.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;- <b>Jnr</b>: no standard distance races required.
    </span>
    """,
    unsafe_allow_html=True
    )   
# with tab3:
#     # Select specific columns
#     data3 = data[['program_id', 'event_id', 'program_date', 'program_name', 'race_level', 
#              'race_distance_calculated', 'event_title', 'athlete_id', 'athlete_title', 'athlete_gender',
#              'athlete_noc','rank', 'total_time', 'splits', 'leg', 'secs_behind']]
#     df_filtered = data3.copy()
    
#     # Athlete select box at the top of tab 3
#     st.markdown("### Race Results Table")
#     if selected_name:
#         filtered_df = df_filtered[df_filtered['athlete_title'] == selected_name]

#         selected_columns = [
#             'event_title', 'program_date', 'program_name', 'race_distance_calculated', 
#             'athlete_title', 'athlete_noc', 'rank', 'total_time',
#             'leg', 'splits', 'secs_behind'
#         ]

#         @st.cache_data
#         def get_pivoted_table(filtered_df, selected_columns):
#             table_data = filtered_df[selected_columns]
#             table_data_wide = table_data.pivot_table(
#                 index=['event_title', 'program_date', 'program_name', 'race_distance_calculated', 'athlete_title', 'athlete_noc', 'rank', 'total_time'],
#                 columns='leg',
#                 values=['splits', 'secs_behind'], 
#                 dropna=False
#             )
#             # Flatten the MultiIndex columns
#             table_data_wide.columns = ['_'.join(col).strip() for col in table_data_wide.columns.values]
#             table_data_wide.reset_index(inplace=True)

#             columns_to_rename = {
#                 'event_title': 'Race',
#                 'program_date': 'Date',
#                 'program_name': 'Program',
#                 'race_distance_calculated': 'Distance',
#                 'athlete_title': 'Athlete',
#                 'athlete_noc': 'Country',
#                 'rank': 'Rank',
#                 'total_time': 'Total Time',
#                 'secs_behind_swim': 'TBF Swim',
#                 'secs_behind_bike': 'TBF Bike',
#                 'secs_behind_run': 'TBF Run',
#                 'splits_swim': 'Swim',
#                 'splits_bike': 'Bike',
#                 'splits_run': 'Run',
#             }

#             table_data_wide2 = table_data_wide.rename(columns=columns_to_rename)

#             # Reordering columns using loc
#             display_columns_table2 = ['Race', 'Date', 'Program', 'Distance', 'Athlete', 'Country', 'Rank', 'Total Time', 'Swim','TBF Swim', 'Bike', 'TBF Bike', 'Run','TBF Run']
#             table_data_wide2 = table_data_wide2.loc[:, display_columns_table2]

#             # Convert 'Total Time' and all columns between 'Total Time' and 'TBF Run' to datetime format and then to string format
#             columns_to_convert = table_data_wide2.loc[:, 'Total Time':'TBF Run'].columns

#             # Fill all NaN values in the time columns with 0 before formatting
#             table_data_wide2[columns_to_convert] = table_data_wide2[columns_to_convert].fillna(0)

#             for col in columns_to_convert:
#                 if pd.api.types.is_numeric_dtype(table_data_wide2[col]):
#                     # Convert numeric seconds to datetime, then to string, replace NaT with "00:00:00"
#                     table_data_wide2[col] = (
#                         pd.to_datetime(table_data_wide2[col], unit='s', errors='coerce')
#                         .dt.strftime('%H:%M:%S')
#                         .replace('NaT', '00:00:00')
#                     )
#                 else:
#                     # If already string or object, just fillna with "00:00:00"
#                     table_data_wide2[col] = table_data_wide2[col].replace(0, '00:00:00').fillna('00:00:00')

#             return table_data_wide2

#         # Call the function to get the pivoted table for the selected athlete
#         table_data_wide2 = get_pivoted_table(filtered_df, selected_columns)

#         # Display the filtered DataFrame
#         st.dataframe(table_data_wide2, use_container_width=True, hide_index=True)
#     else:
#         st.info("To begin, please select an athlete from the dropdown above.")
