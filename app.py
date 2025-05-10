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
import math
import warnings
warnings.filterwarnings('ignore')

# Set page config
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
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    deliveries = pd.read_csv('deliveries_2024 compressed.csv')
    matches = pd.read_csv('matches_till_2024.csv')
    venue_profiles = pd.read_csv('venue_profiles.csv')
    stadium_reports = pd.read_csv('Stadium_Reports.csv')
    
    matches = matches.rename(columns = {'id':'match_id'})
    df = deliveries.merge(matches, on = 'match_id')
    venue_profiles = venue_profiles.rename(columns={'Venue':'venue'})
    df = df.merge(venue_profiles, on='venue', how='outer', suffixes=('', '_venue'))
    df = df.merge(stadium_reports, on='venue', how='outer', suffixes=('', '_stadium'))
    
    return df, deliveries, matches, venue_profiles, stadium_reports

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

# Sidebar for navigation and filtering
def render_sidebar(venues, teams):
    with st.sidebar:
        st.image("https://www.freepnglogos.com/uploads/cricket-logo-png/cricket-logo-svg-vector-download-18.png", width=100)
        st.markdown("## 🔍 Filters & Navigation")
        
        analysis_type = st.radio(
            "Select Analysis Type",
            ["Venue Intelligence", "Team Strategy", "Player Performance", "Prediction Center"]
        )
        
        selected_venue = st.selectbox("Select Venue", sorted(venues))
        
        if analysis_type in ["Team Strategy", "Player Performance", "Prediction Center"]:
            selected_team = st.selectbox("Select Team", sorted(teams))
            opponent_team = st.selectbox("Select Opponent", [t for t in sorted(teams) if t != selected_team])
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
        st.markdown("##### Developed  by  Yash Vardhan")
        
    return analysis_type, selected_venue, selected_team, opponent_team, season_range

# Venue Intelligence Page
def show_venue_intelligence(df, selected_venue, season_range):
    st.markdown(f"## 🏟️ Venue Analysis: {selected_venue}")
    
    venue_data = df[df['venue'] == selected_venue]
    
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
        st.error(f"No data available for {selected_venue}. Please select another venue.")
        return
    
    # Metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("Avg 1st Innings", f"{avg_1st_inn:.1f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("Avg 2nd Innings", f"{avg_2nd_inn:.1f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("Batting 1st Win %", f"{bat_first_win}%")
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("Chasing Win %", f"{chase_win}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Venue characteristics in a table with visual indicators
    st.markdown("### Pitch Characteristics")
    
    # Create a more visual representation of pitch characteristics
    char_col1, char_col2 = st.columns(2)
    
    with char_col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown(f"**Pitch Type:** {pitch_type}")
        st.markdown(f"**Bounce Level:** {bounce}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with char_col2:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown(f"**Spin Assistance:** {spin_assist}")
        st.markdown(f"**Seam Movement:** {seam_move}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Phase-wise analysis
    st.markdown("### Phase-wise Analysis")
    
    # Add phase to dataframe
    venue_data['phase'] = venue_data['over'].apply(assign_phase)
    
    # Calculate phase stats
    phase_stats = venue_data.groupby('phase').agg(
        total_runs=('batsman_runs', 'sum'),
        total_balls=('ball', 'count'),
        total_wickets=('is_wicket', 'sum')
    ).reset_index()
    
    phase_stats['run_rate'] = phase_stats['total_runs'] / (phase_stats['total_balls'] / 6)
    phase_stats['wicket_rate'] = phase_stats['total_wickets'] / (phase_stats['total_balls'] / 6)
    
    # Ensure phases are in correct order
    phase_order = {'Powerplay': 1, 'Middle Overs': 2, 'Death Overs': 3}
    phase_stats['phase_order'] = phase_stats['phase'].map(phase_order)
    phase_stats = phase_stats.sort_values('phase_order')
    
    # Create a bar chart for run rates with plotly
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=phase_stats['phase'],
            y=phase_stats['run_rate'],
            name="Run Rate",
            marker_color='rgb(26, 118, 255)',
            opacity=0.75
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=phase_stats['phase'],
            y=phase_stats['wicket_rate'],
            name="Wicket Rate",
            marker_color='rgb(255, 61, 61)',
            mode='lines+markers'
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        title_text='Run Rate and Wicket Rate by Phase',
        xaxis_title='Phase',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template="plotly_white",
        height=500
    )
    
    fig.update_yaxes(title_text="Run Rate", secondary_y=False, tickformat=".2f")
    fig.update_yaxes(title_text="Wicket Rate", secondary_y=True, tickformat=".2f")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Historical trends tab
    st.markdown("### Historical Performance")
    
    # Get season-wise data for this venue
    season_data = venue_data.groupby('season').agg(
        matches=('match_id', 'nunique'),
        avg_score=('total_runs', lambda x: x.sum() / len(x.unique())),
        total_wickets=('is_wicket', 'sum')
    ).reset_index()
    
    season_data = season_data.sort_values('season')
    
    # Create a line chart for season trends
    fig = px.line(
        season_data, 
        x='season', 
        y='avg_score', 
        markers=True,
        title=f"Average Score by Season at {selected_venue}",
        labels={"season": "Season", "avg_score": "Average Score"},
    )
    
    fig.update_layout(
        xaxis=dict(tickmode='linear'),
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Memorable moments at this venue
    st.markdown("### 📋 Venue Highlights")
    
    # Top 5 highest team totals
    highest_totals = venue_data.groupby(['match_id', 'batting_team']).agg(
        total_score=('total_runs', 'sum')
    ).reset_index().sort_values('total_score', ascending=False).head(5)
    
    # Top 5 lowest defended totals
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("#### 🔝 Highest Team Totals")
        for i, row in highest_totals.iterrows():
            st.markdown(f"**{row['batting_team']}**: {int(row['total_score'])}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("#### 🌟 Best Individual Performances")
        
        # Calculate top individual performances
        top_batters = venue_data.groupby(['match_id', 'batter']).agg(
            runs=('batsman_runs', 'sum')
        ).reset_index().sort_values('runs', ascending=False).head(5)
        
        for i, row in top_batters.iterrows():
            st.markdown(f"**{row['batter']}**: {int(row['runs'])} runs")
        st.markdown("</div>", unsafe_allow_html=True)

# Team Strategy Page
def show_team_strategy(df, selected_venue, selected_team, opponent_team):
    st.markdown(f"## 📊 Team Strategy: {selected_team} vs {opponent_team}")
    st.markdown(f"### At {selected_venue}")
    
    # Filter data for the selected teams at the selected venue
    team_data = df[(df['venue'] == selected_venue) & 
                 ((df['batting_team'] == selected_team) | (df['batting_team'] == opponent_team))]
    
    if team_data.empty:
        st.error(f"No data available for {selected_team} vs {opponent_team} at {selected_venue}")
        return
    
    # Create tabs for different analysis
    tab1, tab2, tab3 = st.tabs(["Head-to-Head", "Batting Strategy", "Bowling Strategy"])
    
    with tab1:
        st.markdown("### Head-to-Head Analysis")
        
        # Calculate head-to-head stats
        h2h_batting = team_data[team_data['batting_team'] == selected_team]
        h2h_bowling = team_data[team_data['batting_team'] == opponent_team]
        
        total_matches = len(set(h2h_batting['match_id']).union(set(h2h_bowling['match_id'])))
        
        if total_matches == 0:
            st.warning(f"No head-to-head matches found between {selected_team} and {opponent_team} at {selected_venue}")
        else:
            # Calculate team scores and win percentage
            selected_team_wins = len(h2h_batting[h2h_batting['winner'] == selected_team]['match_id'].unique())
            opponent_team_wins = len(h2h_bowling[h2h_bowling['winner'] == opponent_team]['match_id'].unique())
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric("Total Matches", f"{total_matches}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric(f"{selected_team} Wins", f"{selected_team_wins} ({int(selected_team_wins/total_matches*100 if total_matches > 0 else 0)}%)")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col3:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric(f"{opponent_team} Wins", f"{opponent_team_wins} ({int(opponent_team_wins/total_matches*100 if total_matches > 0 else 0)}%)")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Create a pie chart for win distribution
            fig = go.Figure(data=[go.Pie(
                labels=[selected_team, opponent_team],
                values=[selected_team_wins, opponent_team_wins],
                hole=.4,
                marker_colors=['rgb(26, 118, 255)', 'rgb(255, 61, 61)']
            )])
            
            fig.update_layout(
                title_text=f"Win Distribution at {selected_venue}",
                annotations=[dict(text='Wins', x=0.5, y=0.5, font_size=20, showarrow=False)],
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Batting Strategy")
        
        # Add phase to the data
        h2h_batting = team_data[team_data['batting_team'] == selected_team].copy()
        if not h2h_batting.empty:
            h2h_batting['phase'] = h2h_batting['over'].apply(assign_phase)
            
            # Phase-wise batting performance
            phase_batting = h2h_batting.groupby('phase').agg(
                runs=('batsman_runs', 'sum'),
                balls=('ball', 'count'),
                wickets=('is_wicket', 'sum')
            ).reset_index()
            
            phase_batting['run_rate'] = phase_batting['runs'] / (phase_batting['balls'] / 6)
            phase_batting['avg'] = phase_batting['runs'] / phase_batting['wickets'].replace(0, 1)
            
            # Ensure correct phase order
            phase_order = {'Powerplay': 1, 'Middle Overs': 2, 'Death Overs': 3}
            phase_batting['phase_order'] = phase_batting['phase'].map(phase_order)
            phase_batting = phase_batting.sort_values('phase_order')
            
            # Create a bar chart for batting run rates
            fig = px.bar(
                phase_batting,
                x='phase',
                y='run_rate',
                color='phase',
                text='run_rate',
                title=f"{selected_team}'s Run Rate by Phase at {selected_venue}",
                labels={"phase": "Phase", "run_rate": "Run Rate"},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', height=400)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Top batsmen for this team at this venue
            top_batsmen = h2h_batting.groupby('batter').agg(
                runs=('batsman_runs', 'sum'),
                balls=('ball', 'count'),
                dismissals=('is_wicket', 'sum'),
                matches=('match_id', 'nunique')
            ).reset_index()
            
            top_batsmen['average'] = top_batsmen['runs'] / top_batsmen['dismissals'].replace(0, 1)
            top_batsmen['strike_rate'] = top_batsmen['runs'] / top_batsmen['balls'] * 100
            
            # Filter batsmen with minimum balls faced
            top_batsmen = top_batsmen[top_batsmen['balls'] >= 30].sort_values('runs', ascending=False).head(8)
            
            st.markdown("#### Top Performers with Bat")
            
            # Create a scatter plot for batting performances
            fig = px.scatter(
                top_batsmen,
                x='strike_rate',
                y='average',
                size='runs',
                color='runs',
                hover_name='batter',
                log_y=True,
                size_max=30,
                title=f"Batting Performance at {selected_venue}",
                labels={"strike_rate": "Strike Rate", "average": "Average", "runs": "Total Runs"}
            )
            
            fig.update_layout(height=500)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No batting data for {selected_team} against {opponent_team} at {selected_venue}")
    
    with tab3:
        st.markdown("### Bowling Strategy")
        
        # Bowling analysis
        h2h_bowling = team_data[team_data['bowling_team'] == selected_team].copy()
        if not h2h_bowling.empty:
            h2h_bowling['phase'] = h2h_bowling['over'].apply(assign_phase)
            
            # Phase-wise bowling performance
            phase_bowling = h2h_bowling.groupby('phase').agg(
                runs=('batsman_runs', 'sum'),
                balls=('ball', 'count'),
                wickets=('is_wicket', 'sum')
            ).reset_index()
            
            phase_bowling['economy'] = phase_bowling['runs'] / (phase_bowling['balls'] / 6)
            phase_bowling['bowling_avg'] = phase_bowling['runs'] / phase_bowling['wickets'].replace(0, 1)
            phase_bowling['bowling_sr'] = phase_bowling['balls'] / phase_bowling['wickets'].replace(0, 1)
            
            # Ensure correct phase order
            phase_order = {'Powerplay': 1, 'Middle Overs': 2, 'Death Overs': 3}
            phase_bowling['phase_order'] = phase_bowling['phase'].map(phase_order)
            phase_bowling = phase_bowling.sort_values('phase_order')
            
            # Create a bar chart for bowling economy
            fig = px.bar(
                phase_bowling,
                x='phase',
                y='economy',
                color='phase',
                text='economy',
                title=f"{selected_team}'s Bowling Economy by Phase at {selected_venue}",
                labels={"phase": "Phase", "economy": "Economy Rate"},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', height=400)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Top bowlers for this team at this venue
            top_bowlers = h2h_bowling.groupby('bowler').agg(
                runs=('batsman_runs', 'sum'),
                balls=('ball', 'count'),
                wickets=('is_wicket', 'sum'),
                matches=('match_id', 'nunique')
            ).reset_index()
            
            top_bowlers['economy'] = top_bowlers['runs'] / (top_bowlers['balls'] / 6)
            top_bowlers['bowling_avg'] = top_bowlers['runs'] / top_bowlers['wickets'].replace(0, 1)
            top_bowlers['bowling_sr'] = top_bowlers['balls'] / top_bowlers['wickets'].replace(0, 1)
            
            # Filter bowlers with minimum balls bowled
            top_bowlers = top_bowlers[top_bowlers['balls'] >= 24].sort_values('wickets', ascending=False).head(8)
            
            st.markdown("#### Top Performers with Ball")
            
            # Create horizontal bar chart for bowlers
            fig = go.Figure(data=[
                go.Bar(
                    y=top_bowlers['bowler'],
                    x=top_bowlers['wickets'],
                    orientation='h',
                    name='Wickets',
                    marker=dict(color='green'),
                    hovertemplate='%{y}: %{x} wickets<br>Economy: %{customdata:.2f}'
                )
            ])
            
            fig.update_layout(
                title=f"Bowling Performance at {selected_venue}",
                xaxis_title="Wickets",
                yaxis_title="Bowler",
                height=500,
                yaxis={'categoryorder':'total ascending'}
            )
            
            fig.update_traces(customdata=top_bowlers['economy'])
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No bowling data for {selected_team} against {opponent_team} at {selected_venue}")

# Player Performance Page
def show_player_performance(df, selected_venue, selected_team, opponent_team):
    st.markdown(f"## 👤 Player Analysis: {selected_team} vs {opponent_team}")
    st.markdown(f"### At {selected_venue}")
    
    # Filter data for the selected venue
    venue_data = df[df['venue'] == selected_venue].copy()
    
    # Create a function to get player recommendations
    def recommend_players():
        # Select batting data for this team at this venue
        batting_df = venue_data[venue_data['batsman_runs'].notnull() & 
                              venue_data['batter'].notnull() & 
                              (venue_data['batting_team'] == selected_team)]
        
        # Venue-Level Aggregation per Player
        venue_stats = (
            batting_df
            .groupby(['batter', 'venue'])
            .agg({
                'batsman_runs': ['sum', 'mean'],
                'ball': 'count',
                'match_id': pd.Series.nunique,
                'season': lambda x: x.nunique(),
            })
            .reset_index()
        )
        
        venue_stats.columns = ['batter', 'venue', 'total_runs', 'avg_runs', 'balls_faced', 'matches_played', 'seasons_played']
        venue_stats['strike_rate'] = venue_stats['total_runs'] / venue_stats['balls_faced'] * 100
        
        # Recent Form (last 5 innings)
        recent_form_df = (
            batting_df
            .groupby(['match_id', 'batter'])['batsman_runs']
            .sum()
            .reset_index()
            .sort_values(['batter', 'match_id'])
        )
        
        # Take last 5 innings per batter
        recent_form_df['match_order'] = recent_form_df.groupby('batter').cumcount(ascending=False)
        recent_5_avg = (
            recent_form_df[recent_form_df['match_order'] < 5]
            .groupby('batter')['batsman_runs']
            .mean()
            .reset_index()
            .rename(columns={'batsman_runs': 'recent_avg'})
        )
        
        # Opponent Matchup Avg
        vs_team_avg = (
            batting_df
            .groupby(['batter', 'bowling_team'])
            .agg({'batsman_runs': 'sum', 'match_id': pd.Series.nunique})
            .reset_index()
        )
        vs_team_avg['vs_team_avg'] = vs_team_avg['batsman_runs'] / vs_team_avg['match_id']
        vs_team_avg = vs_team_avg[['batter', 'bowling_team', 'vs_team_avg']]
        
        # Merge venue averages
        merged = venue_stats.copy()
        
        # Merge recent form
        merged = merged.merge(recent_5_avg, on='batter', how='left')
        
        # Merge matchup vs current opponent
        matchup = vs_team_avg[vs_team_avg['bowling_team'] == opponent_team]
        merged = merged.merge(matchup, on='batter', how='left')
        
        # Fill NA values
        merged.fillna(0, inplace=True)
        
        # Normalized score calculation
        merged['norm_avg_runs'] = merged['avg_runs'] / merged['avg_runs'].max() if merged['avg_runs'].max() > 0 else 0
        merged['norm_recent_avg'] = merged['recent_avg'] / merged['recent_avg'].max() if merged['recent_avg'].max() > 0 else 0
        merged['norm_vs_team_avg'] = merged['vs_team_avg'] / merged['vs_team_avg'].max() if merged['vs_team_avg'].max() > 0 else 0
        
        # Weighted scoring logic
        merged['predicted_score'] = (
            0.4 * merged['norm_avg_runs'] +
            0.3 * merged['norm_recent_avg'] +
            0.3 * merged['norm_vs_team_avg']
        )
        
        # Sort by predicted performance
        merged = merged.sort_values('predicted_score', ascending=False)
        
        return merged
    
    # Get player recommendations
    player_recommendations = recommend_players()
    
    if player_recommendations.empty:
        st.error(f"No player data available for {selected_team} at {selected_venue}")
        return
    
    # Show top performers as cards
    st.markdown("### 🌟 Player Performance Predictions")
    st.markdown(f"Based on historical data at {selected_venue}, venue characteristics, and matchup vs {opponent_team}")
    
    # Filter to players with some meaningful data
    filtered_recommendations = player_recommendations[(player_recommendations['matches_played'] >= 2) &
                                                    (player_recommendations['total_runs'] >= 50)]
    
    # Display top 5 players predicted to perform well
    if not filtered_recommendations.empty:
        top_players = filtered_recommendations.head(5)
        
        # Create a horizontal bar chart for top players
        fig = px.bar(
            top_players,
            y='batter',
            x='predicted_score',
            color='predicted_score',
            orientation='h',
            text='avg_runs',
            title=f"Top 5 Batters for {selected_team} at {selected_venue}",
            labels={"batter": "Player", "predicted_score": "Performance Score", "avg_runs": "Avg Runs"},
            color_continuous_scale=px.colors.sequential.Viridis,
            height=400
        )
        
        fig.update_traces(texttemplate='Avg: %{text:.1f}', textposition='outside')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Player insights in columns
        st.markdown("### 📊 Detailed Player Statistics")
        
        # Create player cards in columns
        cols = st.columns(5)
        for i, (idx, player) in enumerate(top_players.iterrows()):
            with cols[i % 5]:
                st.markdown(f"""
                <div class='metric-container'>
                    <h4>{player['batter']}</h4>
                    <p><b>Avg: </b>{player['avg_runs']:.1f}</p>
                    <p><b>SR: </b>{player['strike_rate']:.1f}</p>
                    <p><b>Recent Form: </b>{player['recent_avg']:.1f}</p>
                    <p><b>vs {opponent_team}: </b>{player['vs_team_avg']:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning(f"Insufficient data for player recommendations for {selected_team} at {selected_venue}")
    
    # Top matchups - batsmen vs bowlers
    st.markdown("### 🥊 Key Matchups")
    
    # Get batting data for this team
    team_batting = venue_data[(venue_data['batting_team'] == selected_team) & 
                            (venue_data['bowling_team'] == opponent_team)].copy()
    
    if not team_batting.empty:
        # Get batsman vs bowler stats
        batsman_vs_bowler = team_batting.groupby(['batter', 'bowler']).agg(
            runs=('batsman_runs', 'sum'),
            balls=('ball', 'count'),
            dismissals=('is_wicket', 'sum')
        ).reset_index()
        
        batsman_vs_bowler['strike_rate'] = batsman_vs_bowler['runs'] / batsman_vs_bowler['balls'] * 100
        
        # Filter for meaningful matchups
        key_matchups = batsman_vs_bowler[(batsman_vs_bowler['balls'] >= 10)].sort_values('balls', ascending=False)
        
        if not key_matchups.empty:
            # Create cards for key matchups
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.markdown("#### Favorable to Batsmen")
                
                favorable_batsmen = key_matchups.sort_values('strike_rate', ascending=False).head(3)
                for i, row in favorable_batsmen.iterrows():
                    st.markdown(f"""
                    <p><b>{row['batter']} vs {row['bowler']}</b><br>
                    {int(row['runs'])} runs off {int(row['balls'])} balls (SR: {row['strike_rate']:.1f})<br>
                    Dismissals: {int(row['dismissals'])}</p>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.markdown("#### Favorable to Bowlers")
                
                favorable_bowlers = key_matchups[(key_matchups['dismissals'] > 0)].sort_values('strike_rate').head(3)
                for i, row in favorable_bowlers.iterrows():
                    st.markdown(f"""
                    <p><b>{row['batter']} vs {row['bowler']}</b><br>
                    {int(row['runs'])} runs off {int(row['balls'])} balls (SR: {row['strike_rate']:.1f})<br>
                    Dismissals: {int(row['dismissals'])}</p>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No significant head-to-head matchups found with sufficient data")
    else:
        st.info(f"No matchup data available between {selected_team} and {opponent_team} at {selected_venue}")

# Prediction Center Page
def show_prediction_center(df, selected_venue, selected_team, opponent_team):
    st.markdown(f"## 🔮 Prediction Center")
    st.markdown(f"### {selected_team} vs {opponent_team} at {selected_venue}")
    
    # Get venue characteristics
    venue_data = df[df['venue'] == selected_venue]
    
    if venue_data.empty:
        st.error(f"No data available for {selected_venue}")
        return
    
    try:
        avg_1st_innings = venue_data['Avg 1st Innings Score'].iloc[0]
        avg_2nd_innings = venue_data['Avg 2nd Innings Score'].iloc[0]
        batting_first_win = venue_data['Batting 1st Win %'].iloc[0]
        chasing_win = venue_data['Chasing Win %'].iloc[0]
    except (IndexError, KeyError):
        st.error(f"Missing venue profile data for {selected_venue}")
        return
    
    # Display current venue stats
    st.markdown("### Venue Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("Avg 1st Innings", f"{avg_1st_innings:.1f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("Avg 2nd Innings", f"{avg_2nd_innings:.1f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("Batting 1st Win %", f"{batting_first_win}%")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col4:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("Chasing Win %", f"{chasing_win}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Create tabs for different predictions
    tab1, tab2, tab3 = st.tabs(["Score Prediction", "Win Probability", "Strategy Recommendation"])
    
    with tab1:
        st.markdown("### 📈 Score Prediction")
        
        # Simple prediction form
        col1, col2 = st.columns(2)
        
        with col1:
            batting_team = st.radio("Batting Team", [selected_team, opponent_team])
            bowling_team = opponent_team if batting_team == selected_team else selected_team
            
        with col2:
            innings = st.radio("Innings", ["1st Innings", "2nd Innings"])
            weather = st.selectbox("Weather Condition", ["Clear", "Cloudy", "Rainy", "Hot and Humid"])
        
        # Get team averages at this venue
        team_venue_data = df[(df['venue'] == selected_venue) & (df['batting_team'] == batting_team)]
        
        if not team_venue_data.empty:
            team_avg_at_venue = team_venue_data.groupby('match_id')['total_runs'].sum().mean()
        else:
            team_avg_at_venue = avg_1st_innings if innings == "1st Innings" else avg_2nd_innings
        
        # Calculate predicted score based on venue average and team average
        base_score = avg_1st_innings if innings == "1st Innings" else avg_2nd_innings
        
        # Weather adjustment
        weather_factor = 1.0
        if weather == "Cloudy":
            weather_factor = 0.95
        elif weather == "Rainy":
            weather_factor = 0.9
        elif weather == "Hot and Humid":
            weather_factor = 1.05
        
        # Team strength adjustment (simplified)
        team_strength_factor = 1.0
        if not team_venue_data.empty:
            team_strength_factor = team_avg_at_venue / base_score
            team_strength_factor = max(0.8, min(team_strength_factor, 1.2))  # Limit the adjustment
        
        # Calculate predicted score
        predicted_score = int(base_score * weather_factor * team_strength_factor)
        
        # Calculate lower and upper bounds (90% confidence interval)
        lower_bound = int(predicted_score * 0.9)
        upper_bound = int(predicted_score * 1.1)
        
        # Display prediction
        st.markdown("#### Predicted Score")
        
        # Create a gauge chart for the prediction
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = predicted_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Predicted Score for {batting_team}"},
            delta = {'reference': base_score, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [None, max(predicted_score * 1.2, avg_1st_innings * 1.2)]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, lower_bound], 'color': "lightgray"},
                    {'range': [lower_bound, upper_bound], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': base_score
                }
            }
        ))
        
        fig.update_layout(height=350)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        <div class='highlight-box'>
            <p>Our model predicts a score of <b>{predicted_score}</b> for {batting_team} in the {innings} 
            (range: {lower_bound}-{upper_bound}).</p>
            <p><b>Key factors:</b> {weather} conditions, team's batting strength, and {selected_venue}'s characteristics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 🏆 Win Probability")
        
        # User inputs for win probability
        col1, col2 = st.columns(2)
        
        with col1:
            toss_winner = st.radio("Toss Winner", [selected_team, opponent_team])
            toss_decision = st.radio("Toss Decision", ["Bat", "Bowl"])
        
        with col2:
            pitch_condition = st.selectbox("Pitch Condition", ["Fresh", "Used", "Dry", "Green"])
            match_importance = st.selectbox("Match Importance", ["League Match", "Must Win", "Knockout"])
        
        # Calculate win probabilities based on toss and venue stats
        base_probability = 50
        
        # Toss advantage
        toss_advantage = 5  # 5% advantage for winning the toss
        
        # Venue advantage
        venue_advantage = 0
        if toss_decision == "Bat" and batting_first_win > chasing_win:
            venue_advantage = (batting_first_win - chasing_win) / 5  # Scaled advantage
        elif toss_decision == "Bowl" and chasing_win > batting_first_win:
            venue_advantage = (chasing_win - batting_first_win) / 5  # Scaled advantage
        
        # Pitch condition adjustment
        pitch_factor = 0
        if pitch_condition == "Fresh" and toss_decision == "Bat":
            pitch_factor = 3
        elif pitch_condition == "Used" and toss_decision == "Bowl":
            pitch_factor = 3
        elif pitch_condition == "Dry" and toss_decision == "Bat":
            pitch_factor = 2
        elif pitch_condition == "Green" and toss_decision == "Bowl":
            pitch_factor = 4
        
        # Match importance factor (psychological)
        importance_factor = 0
        if match_importance == "Must Win":
            importance_factor = 2
        elif match_importance == "Knockout":
            importance_factor = 3
        
        # Calculate probabilities
        toss_winner_prob = base_probability + toss_advantage + venue_advantage + pitch_factor + importance_factor
        toss_loser_prob = 100 - toss_winner_prob
        
        # Assign to teams
        if toss_winner == selected_team:
            team1_prob = toss_winner_prob
            team2_prob = toss_loser_prob
        else:
            team1_prob = toss_loser_prob
            team2_prob = toss_winner_prob
        
        # Ensure probabilities are in valid range
        team1_prob = max(min(team1_prob, 90), 10)
        team2_prob = 100 - team1_prob
        
        # Create a gauge chart for win probability
        fig = go.Figure()
        
        # Add a pie chart for win probability
        fig.add_trace(go.Pie(
            labels=[selected_team, opponent_team],
            values=[team1_prob, team2_prob],
            hole=0.5,
            marker_colors=['rgb(26, 118, 255)', 'rgb(255, 61, 61)'],
            textinfo='label+percent',
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"Match Win Probability",
            annotations=[dict(text='Win %', x=0.5, y=0.5, font_size=15, showarrow=False)],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show reasoning
        st.markdown(f"""
        <div class='highlight-box'>
            <p><b>Win Probability Analysis:</b></p>
            <ul>
                <li><b>Toss Factor:</b> {toss_winner} won the toss and elected to {toss_decision.lower()} first (+{toss_advantage}%)</li>
                <li><b>Venue History:</b> At {selected_venue}, teams batting first win {batting_first_win}% of matches vs {chasing_win}% when chasing (+{venue_advantage:.1f}%)</li>
                <li><b>Pitch Condition:</b> {pitch_condition} pitch favors the {toss_decision.lower()}ing team first (+{pitch_factor}%)</li>
                <li><b>Match Context:</b> {match_importance} adds psychological factor (+{importance_factor}%)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 📋 Strategy Recommendation")
        
        # Get venue data to make strategy recommendations
        pitch_type = venue_data['Pitch Type'].iloc[0] if 'Pitch Type' in venue_data.columns else "Unknown"
        spin_assistance = venue_data['Spin Assistance'].iloc[0] if 'Spin Assistance' in venue_data.columns else "Unknown"
        seam_movement = venue_data['Seam Movement'].iloc[0] if 'Seam Movement' in venue_data.columns else "Unknown"
        
        # Determine recommendations based on venue characteristics
        batting_strategy = ""
        bowling_strategy = ""
        team_composition = ""
        
        # Batting strategy
        if spin_assistance in ["High", "Very High"]:
            batting_strategy = "Play spin cautiously, target pace bowlers. Use feet against spinners."
        elif seam_movement in ["High", "Very High"]:
            batting_strategy = "Be watchful in the powerplay. Target spin in middle overs."
        elif pitch_type == "Flat":
            batting_strategy = "Aggressive from the start. Target big powerplay scores."
        else:
            batting_strategy = "Balanced approach. Assess conditions before acceleration."
        
        # Bowling strategy
        if spin_assistance in ["High", "Very High"]:
            bowling_strategy = "Bowl more spin overs. Use pace as a change-up option."
        elif seam_movement in ["High", "Very High"]:
            bowling_strategy = "Attack with seamers in powerplay. Use cutters and slower balls."
        elif pitch_type == "Flat":
            bowling_strategy = "Defensive lines and changes of pace. Wide yorkers at death."
        else:
            bowling_strategy = "Mixed bowling attack. Use conditions to determine primary attackers."
        
        # Team composition
        if spin_assistance in ["High", "Very High"]:
            team_composition = "Play extra spinners, including part-timers."
        elif seam_movement in ["High", "Very High"]:
            team_composition = "Play extra seam options. Consider batting depth."
        elif pitch_type == "Flat":
            team_composition = "Balanced team with power hitting and bowling variety."
        else:
            team_composition = "Standard balanced side with role clarity."
        
        # Display strategy recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.markdown("#### Batting Strategy")
            st.markdown(f"{batting_strategy}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.markdown("#### Bowling Strategy")
            st.markdown(f"{bowling_strategy}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.markdown("#### Team Composition")
            st.markdown(f"{team_composition}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.markdown("#### Key Match-ups to Target")
            
            # Get player match-ups if data is available
            team_batting = venue_data[(venue_data['batting_team'] == selected_team) & 
                                    (venue_data['bowling_team'] == opponent_team)]
            
            if not team_batting.empty:
                batsman_vs_bowler = team_batting.groupby(['batter', 'bowler']).agg(
                    runs=('batsman_runs', 'sum'),
                    balls=('ball', 'count'),
                    dismissals=('is_wicket', 'sum')
                ).reset_index()
                
                batsman_vs_bowler['strike_rate'] = batsman_vs_bowler['runs'] / batsman_vs_bowler['balls'] * 100
                
                # Get favorable match-ups
                if not batsman_vs_bowler.empty:
                    favorable = batsman_vs_bowler[(batsman_vs_bowler['balls'] >= 10) & 
                                              (batsman_vs_bowler['strike_rate'] >= 150)].head(2)
                    
                    for i, row in favorable.iterrows():
                        st.markdown(f"• {row['batter']} vs {row['bowler']} (SR: {row['strike_rate']:.1f})")
                else:
                    st.markdown("• Insufficient head-to-head data")
            else:
                st.markdown("• Insufficient match-up data")
            
            st.markdown("</div>", unsafe_allow_html=True)

# Main function
def main():
    # Load data
    df, deliveries, matches, venue_profiles, stadium_reports = load_data()
    
    # Get unique venues and teams
    venues = df['venue'].dropna().unique().tolist()
    teams = df['batting_team'].dropna().unique().tolist()
    
    # Render header
    render_header()
    
    # Render sidebar and get selections
    analysis_type, selected_venue, selected_team, opponent_team, season_range = render_sidebar(venues, teams)
    
    # Filter data by season range
    df = df[(df['season'] >= season_range[0]) & (df['season'] <= season_range[1])]
    
    # Display appropriate content based on selection
    if analysis_type == "Venue Intelligence":
        show_venue_intelligence(df, selected_venue, season_range)
    elif analysis_type == "Team Strategy":
        show_team_strategy(df, selected_venue, selected_team, opponent_team)
    elif analysis_type == "Player Performance":
        show_player_performance(df, selected_venue, selected_team, opponent_team)
    elif analysis_type == "Prediction Center":
        show_prediction_center(df, selected_venue, selected_team, opponent_team)

if __name__ == "__main__":
    main()
