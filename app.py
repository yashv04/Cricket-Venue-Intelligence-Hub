import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set page configuration
st.set_page_config(
    page_title="Cricket Venue Intelligence Hub",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load data
@st.cache_data
def load_data():
    deliveries = pd.read_csv('deliveries_2024 compressed.csv')
    matches = pd.read_csv('matches_till_2024.csv')
    venue_profiles = pd.read_csv('venue_profiles.csv')
    stadium_reports = pd.read_csv('Stadium_Reports.csv')
    
    # Rename columns
    matches = matches.rename(columns={'id': 'match_id'})
    venue_profiles = venue_profiles.rename(columns={'Venue': 'venue'})
    
    # Merge datasets
    df = deliveries.merge(matches, on='match_id')
    df = df.merge(venue_profiles, on='venue', how='outer', suffixes=('', '_venue'))
    df = df.merge(stadium_reports, on='venue', how='outer', suffixes=('', '_stadium'))
    
    # Filter innings
    df = df[df['inning'].isin([1, 2])]
    
    # Add phase information
    df['phase'] = df['over'].apply(lambda x: 'Powerplay' if x <= 6 else 
                                  ('Middle Overs' if x <= 15 else 'Death Overs'))
    
    return df, deliveries, matches, venue_profiles, stadium_reports

# Function to create venue summary
def create_venue_summary(df):
    # 1st and 2nd innings analysis
    innings_runs = df.groupby(['venue', 'match_id', 'inning'])['total_runs'].sum().reset_index()
    innings_pivot = innings_runs.pivot(index=['venue', 'match_id'], columns='inning', 
                                      values='total_runs').reset_index()
    innings_pivot.columns = ['venue', 'match_id', 'innings_1_score', 'innings_2_score']
    
    innings_avg = innings_pivot.groupby('venue').agg(
        avg_1st_innings_score=('innings_1_score', 'mean'),
        avg_2nd_innings_score=('innings_2_score', 'mean')
    ).reset_index()
    
    # Main venue summary
    venue_summary = df.groupby('venue').agg(
        total_matches=('match_id', 'nunique'),
        total_runs=('batsman_runs', 'sum'),
        total_wickets=('is_wicket', 'sum'),
        total_balls=('ball', 'count')
    ).reset_index()
    
    venue_summary['avg_runs_per_match'] = venue_summary['total_runs'] / venue_summary['total_matches']
    venue_summary['runs_per_over'] = venue_summary['total_runs'] / (venue_summary['total_balls'] / 6)
    venue_summary['balls_per_wicket'] = venue_summary['total_balls'] / venue_summary['total_wickets']
    
    # Merge with innings averages
    venue_summary = venue_summary.merge(innings_avg, on='venue', how='left')
    
    return venue_summary

# Function to analyze phases by venue
def analyze_phases(df):
    phase_summary = df.groupby(['venue', 'phase']).agg(
        total_runs=('batsman_runs', 'sum'),
        balls=('ball', 'count'),
        wickets=('is_wicket', 'sum')
    ).reset_index()
    
    phase_summary['run_rate'] = phase_summary['total_runs'] / (phase_summary['balls'] / 6)
    phase_summary['strike_rate'] = (phase_summary['total_runs'] / phase_summary['balls']) * 100
    phase_summary['balls_per_wicket'] = phase_summary['balls'] / phase_summary['wickets']
    
    phase_pivot = phase_summary.pivot(index='venue', columns='phase', values='run_rate').reset_index()
    phase_pivot.columns.name = None
    if 'Powerplay' not in phase_pivot.columns:
        phase_pivot['Powerplay'] = 0
    if 'Middle Overs' not in phase_pivot.columns:
        phase_pivot['Middle Overs'] = 0
    if 'Death Overs' not in phase_pivot.columns:
        phase_pivot['Death Overs'] = 0
        
    return phase_summary, phase_pivot

# Function to analyze toss impact
def analyze_toss(df):
    match_toss = df.copy()
    match_toss['bat_first'] = match_toss.apply(lambda x: x['toss_winner']
                                          if x['toss_decision'] == 'bat'
                                          else (x['team1'] if x['toss_winner'] != x['team1'] 
                                               else x['team2']), axis=1)
    
    match_toss['bat_first_won'] = (match_toss['winner'] == match_toss['bat_first']).astype(int)
    
    toss_impact = match_toss.groupby('venue')['bat_first_won'].agg(['count', 'sum']).reset_index()
    toss_impact.columns = ['venue', 'total_matches', 'bat_first_wins']
    toss_impact['bat_first_win_pct'] = toss_impact['bat_first_wins'] / toss_impact['total_matches'] * 100
    toss_impact['bowl_first_win_pct'] = 100 - toss_impact['bat_first_win_pct']
    
    return toss_impact

# Function for player venue analysis
def analyze_player_venue(df, venue=None, min_runs=100):
    player_venue = df.groupby(['venue', 'batter']).agg(
        runs=('batsman_runs', 'sum'),
        balls=('ball', 'count'),
        dismissals=('is_wicket', 'sum'),
        matches=('match_id', 'nunique')
    ).reset_index()
    
    player_venue['strike_rate'] = (player_venue['runs'] / player_venue['balls']) * 100
    player_venue['average'] = player_venue['runs'] / player_venue['dismissals'].replace(0, 1)
    player_venue['runs_per_match'] = player_venue['runs'] / player_venue['matches']
    
    if venue:
        player_venue = player_venue[player_venue['venue'] == venue]
        
    return player_venue[player_venue['runs'] >= min_runs].sort_values('runs', ascending=False)

# Function to analyze batters likely to perform well
def predict_batters(df, venue, opponent):
    # Get venue-level stats
    batting_df = df[df['batsman_runs'].notna() & df['batter'].notna()]
    
    # Handle case where df has no batter column
    if 'batter' not in batting_df.columns:
        return pd.DataFrame()  # Return empty DataFrame if no batter data
    
    # Safe aggregation with error handling
    try:
        venue_stats = batting_df.groupby(['batter', 'venue']).agg({
            'batsman_runs': 'sum',
            'ball': 'count',
            'match_id': 'nunique',
        }).reset_index()
        
        # Add seasons if available
        if 'season' in batting_df.columns:
            seasons = batting_df.groupby(['batter', 'venue'])['season'].nunique().reset_index()
            venue_stats = venue_stats.merge(seasons, on=['batter', 'venue'])
        else:
            venue_stats['season'] = 1
        
        venue_stats.columns = ['batter', 'venue', 'total_runs', 'balls_faced', 'matches_played', 'seasons_played']
        
        # Avoid division by zero
        venue_stats['avg_runs'] = venue_stats['total_runs'] / venue_stats['matches_played'].replace(0, 1)
        venue_stats['strike_rate'] = (venue_stats['total_runs'] / venue_stats['balls_faced'].replace(0, 1)) * 100
        
        # Recent form - simplified to avoid errors
        if 'match_id' in batting_df.columns:
            recent_form_df = batting_df.groupby(['batter', 'match_id'])['batsman_runs'].sum().reset_index()
            # Get the recent average for each batter (last 5 matches if available)
            recent_5_avg = recent_form_df.groupby('batter')['batsman_runs'].mean().reset_index()
            recent_5_avg.columns = ['batter', 'recent_avg']
        else:
            # Create empty DataFrame with required columns
            recent_5_avg = pd.DataFrame(columns=['batter', 'recent_avg'])
        
        # Opponent matchup - simplified
        if 'bowling_team' in batting_df.columns:
            vs_team_avg = batting_df.groupby(['batter', 'bowling_team'])['batsman_runs'].mean().reset_index()
            vs_team_avg.columns = ['batter', 'bowling_team', 'vs_team_avg']
        else:
            # Create empty DataFrame with required columns
            vs_team_avg = pd.DataFrame(columns=['batter', 'bowling_team', 'vs_team_avg'])
    
        # Merge all metrics
        merged = venue_stats[venue_stats['venue'] == venue].copy()
        merged = merged.merge(recent_5_avg, on='batter', how='left')
    
        matchup = vs_team_avg[vs_team_avg['bowling_team'] == opponent]
        merged = merged.merge(matchup, on='batter', how='left')
        
        # Fill NA values
        merged.fillna(0, inplace=True)
        
        # Prediction scoring (with safety checks to prevent division by zero)
        max_avg_runs = merged['avg_runs'].max() if not merged.empty else 0
        max_recent_avg = merged['recent_avg'].max() if 'recent_avg' in merged.columns and not merged.empty else 0
        max_vs_team_avg = merged['vs_team_avg'].max() if 'vs_team_avg' in merged.columns and not merged.empty else 0
        
        merged['norm_avg_runs'] = merged['avg_runs'] / max_avg_runs if max_avg_runs > 0 else 0
        merged['norm_recent_avg'] = merged['recent_avg'] / max_recent_avg if max_recent_avg > 0 else 0
        merged['norm_vs_team_avg'] = merged['vs_team_avg'] / max_vs_team_avg if max_vs_team_avg > 0 else 0
        
        # Ensure all required columns exist
        for col in ['norm_avg_runs', 'norm_recent_avg', 'norm_vs_team_avg']:
            if col not in merged.columns:
                merged[col] = 0
        
        # Weighted scoring
        merged['predicted_score'] = (
            0.4 * merged['norm_avg_runs'] +
            0.3 * merged['norm_recent_avg'] +
            0.3 * merged['norm_vs_team_avg']
        )
        
        # Select required columns, handling cases where some might be missing
        output_columns = ['batter', 'avg_runs', 'total_runs', 'strike_rate', 'predicted_score']
        
        # Add optional columns if they exist
        if 'recent_avg' in merged.columns:
            output_columns.append('recent_avg')
        if 'vs_team_avg' in merged.columns:
            output_columns.append('vs_team_avg')
            
        # Final output
        top_batters = merged.sort_values('predicted_score', ascending=False)[
            [col for col in output_columns if col in merged.columns]
        ]
        
        return top_batters
    except Exception as e:
        # If any error occurs, return empty DataFrame
        print(f"Error in predict_batters: {e}")
        return pd.DataFrame()

# Load data
try:
    df, deliveries, matches, venue_profiles, stadium_reports = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False
    
# Main application
st.title("🏏 Cricket Venue Intelligence Hub")
st.markdown("""
This app provides comprehensive insights about IPL cricket venues, helping analysts, 
team strategists, broadcasters, and fans understand venue characteristics and their impact on match outcomes.
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", 
                        ["Venue Overview", 
                         "Venue Comparison", 
                         "Player Performance",
                         "Predictive Analytics"])

if data_loaded:
    # Get venue list
    venues = sorted(df['venue'].unique())
    teams = sorted(df['team1'].unique())

    # Venue Overview Page
    if page == "Venue Overview":
        st.header("Venue Overview")
        
        selected_venue = st.selectbox("Select a Venue", venues)
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Venue Stats", "Phase Analysis", "Toss Impact"])
        
        with tab1:
            # Get venue summary
            venue_summary = create_venue_summary(df)
            venue_data = venue_summary[venue_summary['venue'] == selected_venue].iloc[0]
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Venue Statistics")
                st.metric("Total Matches", f"{venue_data['total_matches']:.0f}")
                st.metric("Avg Runs per Match", f"{venue_data['avg_runs_per_match']:.1f}")
                st.metric("Runs per Over", f"{venue_data['runs_per_over']:.2f}")
                st.metric("Balls per Wicket", f"{venue_data['balls_per_wicket']:.1f}")
                
            with col2:
                st.subheader("Innings Comparison")
                
                # Create a bar chart for innings comparison
                innings_data = pd.DataFrame({
                    'Innings': ['1st Innings', '2nd Innings'],
                    'Average Score': [venue_data['avg_1st_innings_score'], venue_data['avg_2nd_innings_score']]
                })
                
                fig = px.bar(innings_data, x='Innings', y='Average Score', 
                            title=f"Innings Comparison at {selected_venue}",
                            color='Innings', text_auto='.1f')
                st.plotly_chart(fig, use_container_width=True)
            
            # Venue profile if available
            if 'stadium_reports' in locals():
                try:
                    venue_profile = stadium_reports[stadium_reports['venue'] == selected_venue].iloc[0]
                    st.subheader("Venue Profile")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Pitch Type", venue_profile.get('Pitch Type', 'N/A'))
                        st.metric("Bounce Level", venue_profile.get('Bounce Level', 'N/A'))
                    with col2:
                        st.metric("Spin Assistance", venue_profile.get('Spin Assistance', 'N/A'))
                        st.metric("Seam Movement", venue_profile.get('Seam Movement', 'N/A'))
                    with col3:
                        st.metric("Dew Impact", venue_profile.get('Dew Impact', 'N/A'))
                        st.metric("Ground Shape", venue_profile.get('Ground Shape', 'N/A'))
                except:
                    st.info("Detailed venue profile not available.")
        
        with tab2:
            st.subheader("Phase Analysis")
            
            # Phase analysis
            phase_summary, _ = analyze_phases(df[df['venue'] == selected_venue])
            
            # Create visuals
            fig = px.bar(phase_summary, x='phase', y='run_rate', 
                        title=f"Run Rate by Phase at {selected_venue}",
                        color='phase', text_auto='.2f')
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(phase_summary, x='phase', y='strike_rate', 
                            title="Strike Rate by Phase", color='phase', text_auto='.1f')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(phase_summary, x='phase', y='balls_per_wicket', 
                            title="Balls per Wicket by Phase", color='phase', text_auto='.1f')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Toss Impact Analysis")
            
            # Toss analysis
            toss_impact = analyze_toss(df)
            venue_toss = toss_impact[toss_impact['venue'] == selected_venue]
            
            if not venue_toss.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Batting First Win %", f"{venue_toss['bat_first_win_pct'].iloc[0]:.1f}%")
                    st.metric("Bowling First Win %", f"{venue_toss['bowl_first_win_pct'].iloc[0]:.1f}%")
                
                with col2:
                    # Create a pie chart for toss decision impact
                    fig = px.pie(values=[venue_toss['bat_first_win_pct'].iloc[0], venue_toss['bowl_first_win_pct'].iloc[0]],
                                names=['Bat First', 'Bowl First'],
                                title="Match Win % by Toss Decision")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommended strategy
                st.subheader("Recommended Toss Strategy")
                if venue_toss['bat_first_win_pct'].iloc[0] > venue_toss['bowl_first_win_pct'].iloc[0]:
                    st.success(f"**BAT FIRST** - Higher success rate ({venue_toss['bat_first_win_pct'].iloc[0]:.1f}%)")
                else:
                    st.success(f"**BOWL FIRST** - Higher success rate ({venue_toss['bowl_first_win_pct'].iloc[0]:.1f}%)")
            else:
                st.info("No toss data available for this venue.")

    # Venue Comparison Page
    elif page == "Venue Comparison":
        st.header("Venue Comparison")
        
        # Create phase pivot for clustering
        _, phase_pivot = analyze_phases(df)
        
        # Select venues to compare
        col1, col2 = st.columns(2)
        with col1:
            venue1 = st.selectbox("Select First Venue", venues, index=0)
        with col2:
            venue2 = st.selectbox("Select Second Venue", venues, index=min(1, len(venues)-1))
        
        # Get venue data
        venue_summary = create_venue_summary(df)
        
        # Show comparison
        if venue1 != venue2:
            venue_data1 = venue_summary[venue_summary['venue'] == venue1].iloc[0]
            venue_data2 = venue_summary[venue_summary['venue'] == venue2].iloc[0]
            
            st.subheader("Venue Statistics Comparison")
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame({
                'Metric': ['Runs per Over', 'Avg 1st Innings', 'Avg 2nd Innings', 'Balls per Wicket'],
                venue1: [venue_data1['runs_per_over'], venue_data1['avg_1st_innings_score'], 
                        venue_data1['avg_2nd_innings_score'], venue_data1['balls_per_wicket']],
                venue2: [venue_data2['runs_per_over'], venue_data2['avg_1st_innings_score'], 
                        venue_data2['avg_2nd_innings_score'], venue_data2['balls_per_wicket']]
            })
            
            # Plot comparison
            fig = px.bar(comparison_df, x='Metric', y=[venue1, venue2], barmode='group',
                        title=f"Venue Comparison: {venue1} vs {venue2}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Phase comparison
            st.subheader("Phase-wise Comparison")
            
            # Get phase data for both venues
            phase_data1 = df[df['venue'] == venue1].groupby('phase').agg(
                run_rate=('batsman_runs', lambda x: x.sum() / (len(x)/6))
            ).reset_index()
            
            phase_data2 = df[df['venue'] == venue2].groupby('phase').agg(
                run_rate=('batsman_runs', lambda x: x.sum() / (len(x)/6))
            ).reset_index()
            
            # Rename columns
            phase_data1 = phase_data1.rename(columns={'run_rate': venue1})
            phase_data2 = phase_data2.rename(columns={'run_rate': venue2})
            
            # Merge
            phase_comp = phase_data1.merge(phase_data2, on='phase')
            
            # Plot
            fig = px.line(phase_comp, x='phase', y=[venue1, venue2], markers=True,
                         title="Run Rate by Phase Comparison", 
                         category_orders={"phase": ["Powerplay", "Middle Overs", "Death Overs"]})
            st.plotly_chart(fig, use_container_width=True)
            
            # Similar venues analysis
            st.subheader("Find Similar Venues")
            
            if len(phase_pivot) >= 3:  # Need at least 3 venues for meaningful clustering
                # Prepare for clustering
                X = phase_pivot.set_index('venue')
                if 'Powerplay' in X.columns and 'Middle Overs' in X.columns and 'Death Overs' in X.columns:
                    X = X[['Powerplay', 'Middle Overs', 'Death Overs']]
                    
                    # Scale data
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Perform clustering
                    n_clusters = min(5, len(X))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    X['cluster'] = kmeans.fit_predict(X_scaled)
                    
                    # Get cluster for selected venues
                    try:
                        cluster1 = X.loc[venue1, 'cluster']
                        
                        # Find similar venues (same cluster)
                        similar_venues = X[X['cluster'] == cluster1].index.tolist()
                        similar_venues.remove(venue1)  # Remove the venue itself
                        
                        if similar_venues:
                            st.success(f"Venues similar to {venue1}: {', '.join(similar_venues[:3])}")
                        else:
                            st.info(f"No venues found with similar characteristics to {venue1}")
                    except:
                        st.warning("Could not perform similarity analysis. Insufficient data.")
                else:
                    st.warning("Phase data incomplete for clustering analysis.")
            else:
                st.warning("Not enough venues for similarity analysis.")
        else:
            st.warning("Please select different venues to compare.")

    # Player Performance Page
    elif page == "Player Performance":
        st.header("Player Performance at Venues")
        
        selected_venue = st.selectbox("Select a Venue", venues)
        min_runs = st.slider("Minimum Runs", 50, 500, 100)
        
        # Get player performance at venue
        player_stats = analyze_player_venue(df, selected_venue, min_runs)
        
        if not player_stats.empty:
            st.subheader(f"Top Performers at {selected_venue}")
            
            # Display top performers table
            st.dataframe(player_stats[['batter', 'runs', 'balls', 'strike_rate', 'average', 'matches']]
                        .sort_values('runs', ascending=False)
                        .reset_index(drop=True)
                        .head(10))
            
            # Visualize top 5 batters by runs
            top5 = player_stats.sort_values('runs', ascending=False).head(5)
            
            fig = px.bar(top5, x='batter', y='runs', 
                        title=f"Top 5 Run Scorers at {selected_venue}",
                        color='strike_rate', text_auto='.0f',
                        color_continuous_scale='Viridis')
            fig.update_layout(coloraxis_colorbar_title='Strike Rate')
            st.plotly_chart(fig, use_container_width=True)
            
            # Strike rate vs average scatter plot
            st.subheader("Batting Efficiency Analysis")
            
            fig = px.scatter(player_stats, x='strike_rate', y='average', 
                            size='runs', hover_name='batter', log_y=True,
                            title=f"Strike Rate vs Average at {selected_venue}")
            
            # Add reference lines to create quadrants
            avg_sr = player_stats['strike_rate'].mean()
            avg_avg = player_stats['average'].mean()
            
            fig.add_hline(y=avg_avg, line_dash="dash", line_color="gray")
            fig.add_vline(x=avg_sr, line_dash="dash", line_color="gray")
            
            # Add annotations for quadrants
            fig.add_annotation(x=max(player_stats['strike_rate'])*0.85, y=max(player_stats['average'])*0.85,
                            text="High SR, High Avg<br>Elite", showarrow=False)
            fig.add_annotation(x=min(player_stats['strike_rate'])*1.15, y=max(player_stats['average'])*0.85,
                            text="Low SR, High Avg<br>Anchors", showarrow=False)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No players found with at least {min_runs} runs at {selected_venue}")

    # Predictive Analytics Page
    elif page == "Predictive Analytics":
        st.header("Predictive Analytics")
        
        # Create tabs
        tab1, tab2 = st.tabs(["Batter Performance Prediction", "Match Strategy"])
        
        with tab1:
            st.subheader("Predict Top Batters")
            
            col1, col2 = st.columns(2)
            with col1:
                pred_venue = st.selectbox("Select Venue", venues, key="pred_venue")
            with col2:
                opponent = st.selectbox("Select Opponent Team", teams)
            
            if st.button("Predict Top Batters"):
                with st.spinner("Analyzing player data..."):
                    top_batters = predict_batters(df, pred_venue, opponent)
                    
                    if not top_batters.empty:
                        st.success(f"Top batters predicted to perform well at {pred_venue} against {opponent}")
                        
                        # Display top batters
                        st.dataframe(top_batters[['batter', 'total_runs', 'avg_runs', 'recent_avg', 
                                                'vs_team_avg', 'strike_rate', 'predicted_score']]
                                    .head(10))
                        
                        # Visualize top 5
                        top5 = top_batters.head(5)
                        
                        fig = px.bar(top5, x='batter', y='predicted_score', 
                                    title=f"Top 5 Predicted Performers at {pred_venue} vs {opponent}",
                                    color='predicted_score', text_auto='.2f')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Factor contribution chart
                        factor_df = pd.DataFrame({
                            'Batter': top5['batter'],
                            'Venue History': top5['norm_avg_runs'] * 0.4,
                            'Recent Form': top5['norm_recent_avg'] * 0.3,
                            'vs Opponent': top5['norm_vs_team_avg'] * 0.3
                        })
                        
                        factor_df_melted = pd.melt(factor_df, id_vars=['Batter'], 
                                                var_name='Factor', value_name='Contribution')
                        
                        fig = px.bar(factor_df_melted, x='Batter', y='Contribution', color='Factor',
                                    title="Factor Contribution to Prediction Score", barmode='stack')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough data to make predictions for this venue and opponent.")
        
        with tab2:
            st.subheader("Match Strategy Recommendations")
            
            strategy_venue = st.selectbox("Select Venue", venues, key="strategy_venue")
            
            # Get toss data
            toss_impact = analyze_toss(df)
            venue_toss = toss_impact[toss_impact['venue'] == strategy_venue]
            
            if not venue_toss.empty:
                # Create metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    bat_win_pct = venue_toss['bat_first_win_pct'].iloc[0]
                    bowl_win_pct = venue_toss['bowl_first_win_pct'].iloc[0]
                    
                    st.metric("Batting First Win %", f"{bat_win_pct:.1f}%")
                    st.metric("Bowling First Win %", f"{bowl_win_pct:.1f}%")
                
                with col2:
                    # Create a pie chart for toss decision impact
                    fig = px.pie(values=[bat_win_pct, bowl_win_pct],
                                names=['Bat First', 'Bowl First'],
                                title="Match Win % by Toss Decision")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Get venue summary for target scores
                venue_summary = create_venue_summary(df)
                venue_data = venue_summary[venue_summary['venue'] == strategy_venue]
                
                if not venue_data.empty:
                    avg_1st = venue_data['avg_1st_innings_score'].iloc[0]
                    avg_2nd = venue_data['avg_2nd_innings_score'].iloc[0]
                    
                    st.subheader("Target Score Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg 1st Innings Score", f"{avg_1st:.1f}")
                    with col2:
                        st.metric("Avg 2nd Innings Score", f"{avg_2nd:.1f}")
                    with col3:
                        diff = avg_1st - avg_2nd
                        st.metric("1st vs 2nd Difference", f"{diff:.1f}", 
                                delta=f"{diff:.1f}", delta_color="normal")
                    
                    # Provide strategy recommendations
                    st.subheader("Strategic Recommendations")
                    
                    strategy_tips = []
                    
                    if bat_win_pct > 55:
                        strategy_tips.append(f"**Strongly prefer batting first** ({bat_win_pct:.1f}% win rate)")
                    
                    if bat_win_pct < 45:
                        strategy_tips.append(f"**Strongly prefer bowling first** ({bowl_win_pct:.1f}% win rate)")
                    
                    # Analyze pitch characteristics if available
                    try:
                        venue_profile = stadium_reports[stadium_reports['venue'] == strategy_venue].iloc[0]
                        
                        if 'Spin Assistance' in venue_profile and venue_profile['Spin Assistance'] in ['High', 'Very High']:
                            strategy_tips.append("**Play extra spinners** - Pitch offers significant spin assistance")
                        
                        if 'Seam Movement' in venue_profile and venue_profile['Seam Movement'] in ['High', 'Very High']:
                            strategy_tips.append("**Play quality pace bowlers** - Pitch offers good seam movement")
                        
                        if 'Dew Impact' in venue_profile and venue_profile['Dew Impact'] in ['High', 'Very High']:
                            strategy_tips.append("**Consider dew factor** - High impact on second innings bowling")
                            
                    except:
                        pass
                    
                    # Add phase-specific strategies
                    phase_data = df[df['venue'] == strategy_venue].groupby('phase').agg(
                        run_rate=('batsman_runs', lambda x: x.sum() / (len(x)/6))
                    ).reset_index()
                    
                    if not phase_data.empty:
                        # Find highest scoring phase
                        highest_rr_phase = phase_data.loc[phase_data['run_rate'].idxmax()]
                        if highest_rr_phase['phase'] == 'Powerplay':
                            strategy_tips.append("**Capitalize on powerplay** - Historically high scoring phase at this venue")
                        elif highest_rr_phase['phase'] == 'Death Overs':
                            strategy_tips.append("**Strong death bowling crucial** - Death overs yield most runs at this venue")
                    
                    for tip in strategy_tips:
                        st.markdown(tip)
                else:
                    st.warning("Not enough data for this venue to provide strategy recommendations.")
            else:
                st.warning("Not enough toss data available for this venue.")

# Main program
if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Cricket Venue Intelligence Hub**  
    Developed by: Yash Vardhan  
    Data last updated: May 2025
    """)
    
    # Handle file upload option for future matches
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload match data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        st.sidebar.success("File uploaded successfully! (Processing not implemented in this demo)")
