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
        st.markdown("##### Developed by Yash Vardhan")
        
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
    filtered_recommendations = player_recommendations[(player_recommendations['matches_played'] > 1)]
    
    if not filtered_recommendations.empty:
        # Display top 5 players
        top_players = filtered_recommendations.head(5)
        
        # Create cards for top players
        cols = st.columns(5)
        for i, (_, player) in enumerate(top_players.iterrows()):
            with cols[i]:
                st.markdown(f"""
                <div class='metric-container'>
                    <h3>{player['batter']}</h3>
                    <p><b>Performance Score:</b> {player['predicted_score']:.2f}</p>
                    <p><b>Avg at Venue:</b> {player['avg_runs']:.1f}</p>
                    <p><b>SR at Venue:</b> {player['strike_rate']:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Create a bar chart to compare players
        fig = px.bar(
            top_players, 
            x='batter', 
            y='predicted_score',
            color='predicted_score',
            labels={'batter': 'Player', 'predicted_score': 'Performance Score'},
            title=f"Top 5 Players for {selected_team} at {selected_venue}",
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Player vs Player Matchups
        st.markdown("### ⚔️ Key Player Matchups")
        
        # Filter data for key bowlers from opponent team
        bowling_df = venue_data[venue_data['bowling_team'] == opponent_team]
        top_bowlers = bowling_df.groupby('bowler').agg(
            wickets=('is_wicket', 'sum'),
            balls_bowled=('ball', 'count')
        ).reset_index()
        
        top_bowlers = top_bowlers[top_bowlers['balls_bowled'] > 24].sort_values('wickets', ascending=False).head(3)
        
        if not top_bowlers.empty:
            # Get head-to-head stats
            batters = filtered_recommendations['batter'].head(3).tolist()
            bowlers = top_bowlers['bowler'].tolist()
            
            # Create matchup matrix
            matchup_data = []
            
            for batter in batters:
                for bowler in bowlers:
                    h2h = venue_data[(venue_data['batter'] == batter) & (venue_data['bowler'] == bowler)]
                    
                    if not h2h.empty:
                        runs = h2h['batsman_runs'].sum()
                        balls = len(h2h)
                        dismissals = h2h['is_wicket'].sum()
                        sr = (runs / balls * 100) if balls > 0 else 0
                        
                        matchup_data.append({
                            'Batter': batter,
                            'Bowler': bowler,
                            'Runs': runs,
                            'Balls': balls,
                            'Dismissals': dismissals,
                            'Strike Rate': sr
                        })
            
            if matchup_data:
                matchup_df = pd.DataFrame(matchup_data)
                
                # Create heatmap
                heatmap_df = matchup_df.pivot(index='Batter', columns='Bowler', values='Strike Rate')
                
                fig = px.imshow(
                    heatmap_df,
                    labels=dict(x="Bowler", y="Batter", color="Strike Rate"),
                    x=heatmap_df.columns,
                    y=heatmap_df.index,
                    color_continuous_scale='RdYlGn',
                    aspect="auto",
                    title="Batter vs Bowler Strike Rate"
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No head-to-head data available for key matchups at this venue")
        else:
            st.info(f"No bowling data for {opponent_team} at {selected_venue}")
    else:
        st.warning(f"Not enough player data for {selected_team} at {selected_venue}")

# Prediction Center Page
def show_prediction_center(df, selected_venue, selected_team, opponent_team, venue_profiles, stadium_reports):
    st.markdown(f"## 🔮 Prediction Center: {selected_venue}")
    
    # Get venue data
    venue_data = df[df['venue'] == selected_venue]
    
    if venue_data.empty:
        st.error(f"No data available for {selected_venue}")
        return
    
    # Create tabs for different predictions
    tab1, tab2 = st.tabs(["1st Innings Score Prediction", "Match Win Probability"])
    
    with tab1:
        st.markdown("### 1st Innings Score Prediction")
        
        # Get venue characteristics from the venue_profiles and stadium_reports
        venue_info = venue_profiles[venue_profiles['venue'] == selected_venue]
        stadium_info = stadium_reports[stadium_reports['venue'] == selected_venue]
        
        if venue_info.empty or stadium_info.empty:
            st.warning("Missing venue data for prediction model")
        else:
            # Create a form for prediction input
            with st.form("score_prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Show the batting team
                    st.markdown(f"**Batting Team**: {selected_team}")
                    
                    # Weather condition
                    weather = st.selectbox(
                        "Weather Condition",
                        options=["Clear", "Cloudy", "Humid", "Rainy"],
                        index=0
                    )
                    
                    # Toss winner
                    toss_winner = st.radio(
                        "Toss Winner",
                        options=[selected_team, opponent_team],
                        horizontal=True
                    )
                
                with col2:
                    # Show the bowling team
                    st.markdown(f"**Bowling Team**: {opponent_team}")
                    
                    # Dew factor
                    dew_factor = st.slider(
                        "Dew Factor (0-10)",
                        min_value=0,
                        max_value=10,
                        value=5
                    )
                    
                    # Pitch freshness
                    pitch_freshness = st.select_slider(
                        "Pitch Condition",
                        options=["New", "Used Once", "Used Multiple Times"],
                        value="Used Once"
                    )
                
                submitted = st.form_submit_button("Predict Score")
                
                if submitted:
                    # In a real model, we'd use these inputs with our RandomForestRegressor
                    # Here we'll simulate a prediction based on venue averages with some randomness
                    
                    # Get the average first innings score for this venue
                    avg_score = venue_data['Avg 1st Innings Score'].iloc[0]
                    
                    # Adjust based on inputs (simulating model behavior)
                    # Weather adjustment
                    weather_factor = {
                        "Clear": 1.0,
                        "Cloudy": 0.95,
                        "Humid": 1.05,
                        "Rainy": 0.9
                    }
                    
                    # Dew adjustment
                    dew_adjustment = 1.0 + (dew_factor - 5) * 0.01
                    
                    # Pitch freshness adjustment
                    pitch_factor = {
                        "New": 1.05,
                        "Used Once": 1.0,
                        "Used Multiple Times": 0.95
                    }
                    
                    # Toss advantage
                    toss_adjustment = 1.02 if toss_winner == selected_team else 0.98
                    
                    # Calculate predicted score with some randomness
                    base_prediction = avg_score * weather_factor[weather] * dew_adjustment * pitch_factor[pitch_freshness] * toss_adjustment
                    
                    # Add some randomness (±15 runs)
                    random_factor = np.random.normal(0, 5)
                    final_prediction = int(base_prediction + random_factor)
                    
                    # Display the prediction
                    st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
                    st.markdown(f"### Predicted 1st Innings Score: {final_prediction}")
                    
                    # Add some insights
                    if final_prediction > avg_score + 10:
                        st.markdown(f"This is **above** the venue average of {int(avg_score)}")
                    elif final_prediction < avg_score - 10:
                        st.markdown(f"This is **below** the venue average of {int(avg_score)}")
                    else:
                        st.markdown(f"This is **close to** the venue average of {int(avg_score)}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Show historical score distribution
                    innings_scores = df[(df['venue'] == selected_venue) & 
                                     (df['inning'] == 1)].groupby('match_id')['total_runs'].sum()
                    
                    fig = px.histogram(
                        innings_scores,
                        nbins=20,
                        title=f"Historical 1st Innings Score Distribution at {selected_venue}",
                        labels={'value': 'Score', 'count': 'Frequency'},
                    )
                    
                    # Add a vertical line for the prediction
                    fig.add_vline(x=final_prediction, line_dash="dash", line_color="red",
                                annotation_text="Prediction", annotation_position="top right")
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Match Win Probability")
        
        # Create a form for win probability prediction
        with st.form("win_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                batting_first = st.radio(
                    "Team Batting First",
                    options=[selected_team, opponent_team],
                    horizontal=True
                )
                
                target_score = st.number_input(
                    "Target Score (if known)",
                    min_value=0,
                    max_value=250,
                    value=0,
                    help="Leave as 0 if predicting before first innings"
                )
            
            with col2:
                toss_winner = st.radio(
                    "Toss Winner",
                    options=[selected_team, opponent_team],
                    horizontal=True
                )
                
                match_time = st.select_slider(
                    "Match Time",
                    options=["Day", "Day-Night", "Night"],
                    value="Day-Night"
                )
            
            submitted = st.form_submit_button("Calculate Win Probability")
            
            if submitted:
                # In a real model, we'd use these inputs with our classifier
                # Here we'll simulate win probabilities based on historical data
                
                # Get batting first and chasing win percentages
                batting_first_win_pct = venue_data['Batting 1st Win %'].iloc[0]
                chasing_win_pct = venue_data['Chasing Win %'].iloc[0]
                
                # Determine base win probability for team of interest (selected_team)
                if batting_first == selected_team:
                    base_prob = batting_first_win_pct / 100
                else:
                    base_prob = chasing_win_pct / 100
                
                # Toss factor (winning toss gives advantage)
                toss_factor = 1.1 if toss_winner == selected_team else 0.9
                
                # Time of day factor
                time_factor = {
                    "Day": 1.0,
                    "Day-Night": 0.9 if batting_first == selected_team else 1.1,  # Advantage to chasing team in day-night
                    "Night": 0.85 if batting_first == selected_team else 1.15  # Stronger advantage to chasing at night
                }
                
                # Calculate win probability
                win_prob = base_prob * toss_factor * time_factor[match_time]
                
                # Adjust if we know the target score
                if target_score > 0:
                    avg_score = venue_data['Avg 1st Innings Score'].iloc[0]
                    
                    if batting_first == selected_team:
                        # If selected team batted first, higher score = better chances
                        score_factor = 1.0 + (target_score - avg_score) / avg_score * 0.5
                    else:
                        # If selected team is chasing, higher target = worse chances
                        score_factor = 1.0 - (target_score - avg_score) / avg_score * 0.5
                    
                    win_prob *= score_factor
                
                # Ensure probability is between 0 and 1
                win_prob = max(0.05, min(0.95, win_prob))
                
                # Display the prediction
                st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
                
                # Create a gauge chart for win probability
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = win_prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"{selected_team} Win Probability"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "rgba(50, 168, 82, 0.8)"},
                        'steps': [
                            {'range': [0, 25], 'color': "rgba(214, 39, 40, 0.6)"},
                            {'range': [25, 50], 'color': "rgba(255, 127, 14, 0.6)"},
                            {'range': [50, 75], 'color': "rgba(44, 160, 44, 0.6)"},
                            {'range': [75, 100], 'color': "rgba(44, 160, 44, 0.8)"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Add some insights
                if win_prob > 0.6:
                    st.markdown(f"{selected_team} is favored to win based on the conditions")
                elif win_prob < 0.4:
                    st.markdown(f"{opponent_team} is favored to win based on the conditions")
                else:
                    st.markdown("This looks like a closely contested match")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show historical win percentage by innings
                first_innings_wins = len(venue_data[(venue_data['inning'] == 1) & 
                                               (venue_data['batting_team'] == venue_data['winner'])]['match_id'].unique())
                second_innings_wins = len(venue_data[(venue_data['inning'] == 2) & 
                                                (venue_data['batting_team'] == venue_data['winner'])]['match_id'].unique())
                
                total_matches = len(venue_data['match_id'].unique())
                
                if total_matches > 0:
                    first_innings_win_pct = first_innings_wins / total_matches * 100
                    second_innings_win_pct = second_innings_wins / total_matches * 100
                    
                    # Create a bar chart for historical win percentages
                    win_data = pd.DataFrame({
                        'Batting': ['First Innings', 'Second Innings'],
                        'Win %': [first_innings_win_pct, second_innings_win_pct]
                    })
                    
                    fig = px.bar(
                        win_data,
                        x='Batting',
                        y='Win %',
                        color='Batting',
                        title=f"Historical Win % by Innings at {selected_venue}",
                        labels={'Batting': 'Batting Innings', 'Win %': 'Win Percentage (%)'},
                        color_discrete_sequence=['rgb(26, 118, 255)', 'rgb(44, 160, 44)']
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

def main():
    # Load data
    df, deliveries, matches, venue_profiles, stadium_reports = load_data()
    
    # Get unique venues and teams
    venues = df['venue'].dropna().unique().tolist()
    teams = df['batting_team'].dropna().unique().tolist()
    
    # Render header
    render_header()
    
    # Render sidebar
    analysis_type, selected_venue, selected_team, opponent_team, season_range = render_sidebar(venues, teams)
    
    # Filter data based on season range if it exists in the data
    if 'season' in df.columns:
        # Convert season to numeric if it's not already
        if df['season'].dtype == 'object':
            df['season'] = pd.to_numeric(df['season'], errors='coerce')
            
        # Now filter
        df_filtered = df[df['season'].between(season_range[0], season_range[1], inclusive='both')]
    else:
        df_filtered = df
    
    # Render selected analysis
    if analysis_type == "Venue Intelligence":
        show_venue_intelligence(df_filtered, selected_venue, season_range)
    elif analysis_type == "Team Strategy":
        show_team_strategy(df_filtered, selected_venue, selected_team, opponent_team)
    elif analysis_type == "Player Performance":
        show_player_performance(df_filtered, selected_venue, selected_team, opponent_team)
    elif analysis_type == "Prediction Center":
        show_prediction_center(df_filtered, selected_venue, selected_team, opponent_team, venue_profiles, stadium_reports)

# Run the app
if __name__ == "__main__":
    main()
