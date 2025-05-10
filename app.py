import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
import math
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Set page config immediately
st.set_page_config(
    page_title="Cricket Venue Intelligence Hub",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visualization
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        background-image: linear-gradient(to bottom right, #f5f7f9, #e9ecef);
    }
    .highlight-box {
        background-color: #ffffff;
        border-left: 5px solid #1e88e5;
        padding: 10px;
        border-radius: 4px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .metric-container {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .venue-header {
        text-align: center; 
        color: #0e1117;
        padding: 20px 0;
        border-bottom: 1px solid #e6e6e6;
        margin-bottom: 20px;
    }
    .css-1v3fvcr { 
        background-color: transparent;
    }
    /* Custom styling for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e88e5 !important;
        color: white !important;
    }
    /* Card styling */
    .player-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .player-card:hover {
        transform: translateY(-5px);
    }
    .player-card img {
        border-radius: 50%;
        margin-bottom: 10px;
    }
    .player-metrics {
        display: flex;
        justify-content: space-around;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load data function with error handling
@st.cache_data
def load_data():
    try:
        # For demonstration, we'll create sample dataframes similar to the ones in your code
        # In a real implementation, you would replace this with your actual file loading code
        
        # Create sample data based on the column names in your code
        # This is just for demonstration when files aren't available
        
        # Example venue data
        venues = ['Wankhede Stadium', 'M Chinnaswamy Stadium', 'Eden Gardens', 'Feroz Shah Kotla', 
                 'MA Chidambaram Stadium', 'Punjab Cricket Association Stadium', 'Rajiv Gandhi International Stadium']
        
        # Example teams
        teams = ['Mumbai Indians', 'Royal Challengers Bangalore', 'Chennai Super Kings', 'Kolkata Knight Riders',
                'Delhi Capitals', 'Punjab Kings', 'Rajasthan Royals', 'Sunrisers Hyderabad', 'Gujarat Titans', 'Lucknow Super Giants']
        
        # Create sample venue profiles
        venue_profiles = pd.DataFrame({
            'venue': venues,
            'Pitch Type': np.random.choice(['Batting', 'Bowling', 'Balanced'], size=len(venues)),
            'Bounce Level': np.random.choice(['Low', 'Medium', 'High'], size=len(venues)),
            'Spin Assistance': np.random.choice(['Low', 'Medium', 'High'], size=len(venues)),
            'Seam Movement': np.random.choice(['Low', 'Medium', 'High'], size=len(venues))
        })
        
        # Create sample stadium reports
        stadium_reports = pd.DataFrame({
            'venue': venues,
            'Batting 1st Win %': np.random.randint(40, 60, size=len(venues)),
            'Chasing Win %': np.random.randint(40, 60, size=len(venues)),
            'Avg 1st Innings Score': np.random.randint(150, 200, size=len(venues)),
            'Avg 2nd Innings Score': np.random.randint(140, 190, size=len(venues)),
            'Dew Impact': np.random.choice(['Low', 'Medium', 'High'], size=len(venues)),
            'Ground Shape': np.random.choice(['Round', 'Oval', 'Rectangle'], size=len(venues)),
            'Weather Condition': np.random.choice(['Sunny', 'Cloudy', 'Humid'], size=len(venues)),
        })
        
        # Create sample matches
        num_matches = 100
        matches = pd.DataFrame({
            'match_id': range(1, num_matches + 1),
            'venue': np.random.choice(venues, size=num_matches),
            'team1': [np.random.choice(teams) for _ in range(num_matches)],
            'team2': [np.random.choice(teams) for _ in range(num_matches)],
            'toss_winner': [np.random.choice(teams) for _ in range(num_matches)],
            'toss_decision': np.random.choice(['bat', 'field'], size=num_matches),
            'winner': [np.random.choice(teams) for _ in range(num_matches)],
            'season': np.random.randint(2008, 2025, size=num_matches),
            'date': pd.date_range(start='2023-01-01', periods=num_matches)
        })
        
        # Create sample deliveries
        players = ['MS Dhoni', 'Virat Kohli', 'Rohit Sharma', 'KL Rahul', 'Rishabh Pant', 'Jasprit Bumrah',
                  'Mohammed Shami', 'Ravindra Jadeja', 'R Ashwin', 'Andre Russell', 'Jos Buttler', 'David Warner', 
                  'Kane Williamson', 'AB de Villiers', 'Faf du Plessis', 'Quinton de Kock', 'Kagiso Rabada']
        
        # Create many deliveries for our matches
        num_deliveries = 10000
        deliveries = pd.DataFrame({
            'match_id': np.random.choice(matches['match_id'], size=num_deliveries),
            'inning': np.random.choice([1, 2], size=num_deliveries),
            'over': np.random.randint(0, 20, size=num_deliveries),
            'ball': np.random.randint(1, 7, size=num_deliveries),
            'batsman_runs': np.random.choice([0, 1, 2, 4, 6], size=num_deliveries, p=[0.4, 0.3, 0.1, 0.15, 0.05]),
            'is_wicket': np.random.choice([0, 1], size=num_deliveries, p=[0.95, 0.05]),
            'batter': np.random.choice(players, size=num_deliveries),
            'bowler': np.random.choice(players, size=num_deliveries),
            'total_runs': np.random.choice([0, 1, 2, 4, 6], size=num_deliveries, p=[0.35, 0.3, 0.1, 0.15, 0.1])
        })
        
        # Merge data to create the main dataframe
        df = deliveries.copy()
        
        # Add batting_team and bowling_team columns
        for match_id in df['match_id'].unique():
            match_info = matches[matches['match_id'] == match_id].iloc[0]
            match_indices = df['match_id'] == match_id
            
            # For inning 1
            inn1_indices = match_indices & (df['inning'] == 1)
            # Randomly assign batting team for inning 1
            batting_team_inn1 = match_info['team1'] if np.random.random() > 0.5 else match_info['team2']
            bowling_team_inn1 = match_info['team2'] if batting_team_inn1 == match_info['team1'] else match_info['team1']
            df.loc[inn1_indices, 'batting_team'] = batting_team_inn1
            df.loc[inn1_indices, 'bowling_team'] = bowling_team_inn1
            
            # For inning 2
            inn2_indices = match_indices & (df['inning'] == 2)
            df.loc[inn2_indices, 'batting_team'] = bowling_team_inn1
            df.loc[inn2_indices, 'bowling_team'] = batting_team_inn1
        
        # Add venue information
        df = pd.merge(df, matches[['match_id', 'venue', 'season', 'date', 'winner']], on='match_id')
        
        # Add venue profiles
        df = df.merge(venue_profiles, on='venue', how='left')
        
        # Add stadium reports
        df = df.merge(stadium_reports, on='venue', how='left')
        
        # Add phase information
        df['phase'] = df['over'].apply(lambda over: 'Powerplay' if over <= 6 else ('Middle Overs' if over <= 15 else 'Death Overs'))
        
        # Try to load actual data if files exist
        try:
            real_deliveries = pd.read_csv('deliveries_till_2024.csv')
            real_matches = pd.read_csv('matches_till_2024.csv')
            real_venue_profiles = pd.read_csv('venue_profiles.csv')
            real_stadium_reports = pd.read_csv('Stadium_Reports.csv')
            
            real_matches = real_matches.rename(columns = {'id':'match_id'})
            real_df = real_deliveries.merge(real_matches, on = 'match_id')
            real_venue_profiles = real_venue_profiles.rename(columns={'Venue':'venue'})
            real_df = real_df.merge(real_venue_profiles, on='venue', how='outer', suffixes=('', '_venue'))
            real_df = real_df.merge(real_stadium_reports, on='venue', how='outer', suffixes=('', '_stadium'))
            
            # Add phase information
            real_df['phase'] = real_df['over'].apply(lambda over: 'Powerplay' if over <= 6 else 
                                                  ('Middle Overs' if over <= 15 else 'Death Overs'))
            
            # Use real data if available
            return real_df, real_deliveries, real_matches, real_venue_profiles, real_stadium_reports
        except:
            # Use synthetic data
            return df, deliveries, matches, venue_profiles, stadium_reports
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return empty dataframes as fallback
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Phase classification function
def assign_phase(over):
    if over <= 6:
        return 'Powerplay'
    elif over <= 15:
        return 'Middle Overs'
    else:
        return 'Death Overs'

# Title and introduction
def render_header():
    st.markdown("<h1 class='venue-header'>🏏 Cricket Venue Intelligence Hub</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        <div class="highlight-box">
            <p>Welcome to the ultimate IPL venue analysis platform. Discover deep insights into stadium characteristics, 
            match outcomes, and player performances across different venues. Make data-driven decisions for 
            team strategies, fantasy cricket, and match predictions.</p>
        </div>
        """, unsafe_allow_html=True)

# Sidebar for navigation and filtering - handles empty dataframes gracefully
def render_sidebar(df):
    with st.sidebar:
        st.image("https://www.freepnglogos.com/uploads/cricket-logo-png/cricket-logo-svg-vector-download-18.png", width=100)
        st.markdown("## 🔍 Filters & Navigation")
        
        # Get unique venues and teams safely
        venues = sorted(df['venue'].dropna().unique().tolist()) if not df.empty else ["No venues available"]
        teams = sorted(df['batting_team'].dropna().unique().tolist()) if not df.empty else ["No teams available"]
        
        analysis_type = st.radio(
            "Select Analysis Type",
            ["Venue Intelligence", "Team Strategy", "Player Performance", "Prediction Center"]
        )
        
        selected_venue = st.selectbox("Select Venue", venues)
        
        if analysis_type in ["Team Strategy", "Player Performance", "Prediction Center"]:
            selected_team = st.selectbox("Select Team", teams)
            opponent_options = [t for t in teams if t != selected_team] if len(teams) > 1 else teams
            opponent_team = st.selectbox("Select Opponent", opponent_options)
        else:
            selected_team = None
            opponent_team = None
            
        st.markdown("---")
        st.markdown("### 🔄 Season Filter")
        min_year, max_year = 2008, 2024
        season_range = st.slider("Select Seasons", min_year, max_year, (min_year, max_year))
        
        st.markdown("---")
        st.markdown("### 📊 Data Source")
        st.info("Data updated until IPL 2024")
        
        # Add a separator
        st.markdown("---")
        st.markdown("##### Developed by Yash Vardhan")
        
    return analysis_type, selected_venue, selected_team, opponent_team, season_range

# Venue Intelligence Page
def show_venue_intelligence(df, selected_venue, season_range):
    st.markdown(f"## 🏟️ Venue Analysis: {selected_venue}")
    
    venue_data = df[df['venue'] == selected_venue]
    
    if venue_data.empty:
        st.error(f"No data available for {selected_venue}. Please select another venue.")
        return
    
    # Get venue details
    try:
        pitch_type = venue_data['Pitch Type'].iloc[0]
        bounce = venue_data['Bounce Level'].iloc[0]
        spin_assist = venue_data['Spin Assistance'].iloc[0]
        seam_move = venue_data['Seam Movement'].iloc[0]
        avg_1st_inn = venue_data['Avg 1st Innings Score'].iloc[0]
        avg_2nd_inn = venue_data['Avg 2nd Innings Score'].iloc[0]
        bat_first_win = venue_data['Batting 1st Win %'].iloc[0]
        chase_win = venue_data['Chasing Win %'].iloc[0]
    except (IndexError, KeyError):
        # Use calculated values if columns not available
        try:
            # Calculate innings averages
            innings_avg = venue_data.groupby('inning')['total_runs'].sum() / venue_data.groupby('inning')['match_id'].nunique()
            avg_1st_inn = innings_avg.get(1, 160)  # Default to 160 if not available
            avg_2n
