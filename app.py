import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Cricket Venue Intelligence Hub",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance the app's appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1e3799;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #3867d6;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .stat-card {
        background-color: #f7f9fc;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .highlight {
        color: #eb2f06;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #1e3799;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stSelectbox label, .stMultiselect label {
        color: #3867d6;
        font-weight: 600;
    }
    div[data-testid="stSidebarNav"] {
        background-image: linear-gradient(#1e3799, #4a69bd);
        padding-top: 2rem;
        padding-bottom: 2rem;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        color: #3867d6;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(62, 103, 214, 0.1);
        border-bottom-color: #3867d6;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and prepare all the necessary data for the app"""
    try:
        deliveries = pd.read_csv('deliveries_2024 compressed.csv')
        matches = pd.read_csv('matches_till_2024.csv')
        venue_profiles = pd.read_csv('venue_profiles.csv')
        stadium_reports = pd.read_csv('Stadium_Reports.csv')
        
        # Preprocessing
        matches = matches.rename(columns={'id': 'match_id'})
        venue_profiles = venue_profiles.rename(columns={'Venue': 'venue'})
        
        # Merge datasets
        df = deliveries.merge(matches, on='match_id')
        df = df.merge(venue_profiles, on='venue', how='outer', suffixes=('', '_venue'))
        df = df.merge(stadium_reports, on='venue', how='outer', suffixes=('', '_stadium'))
        
        return df, deliveries, matches, venue_profiles, stadium_reports
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

def create_venue_summary(df):
    """Create venue summary statistics"""
    # Filter innings 1 and 2
    df = df[df['inning'].isin([1, 2])]
    
    # Phase classification
    df['phase'] = df['over'].apply(lambda over: 
                                'Powerplay' if over <= 6 else 
                                'Middle Overs' if over <= 15 else 'Death Overs')
    
    # Calculate 1st and 2nd innings scores
    innings_runs = df.groupby(['venue', 'match_id', 'inning'])['total_runs'].sum().reset_index()
    innings_pivot = innings_runs.pivot(index=['venue', 'match_id'], 
                                    columns='inning', 
                                    values='total_runs').reset_index()
    
    innings_pivot.columns = ['venue', 'match_id', 'innings_1_score', 'innings_2_score']
    
    # Average scores per venue by innings
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
    
    return venue_summary, df

def create_phase_summary(df):
    """Create phase-wise summary by venue"""
    phase_summary = df.groupby(['venue', 'phase']).agg(
        total_runs=('batsman_runs', 'sum'),
        balls=('ball', 'count'),
        wickets=('is_wicket', 'sum')
    ).reset_index()
    
    phase_summary['run_rate'] = phase_summary['total_runs'] / (phase_summary['balls'] / 6)
    phase_summary['strike_rate'] = (phase_summary['total_runs'] / phase_summary['balls']) * 100
    phase_summary['balls_per_wicket'] = phase_summary['balls'] / phase_summary['wickets']
    
    # Create pivot table for dashboard visualization
    phase_pivot = phase_summary.pivot(index='venue', columns='phase', values='run_rate').reset_index()
    
    return phase_summary, phase_pivot

def analyze_toss_impact(df):
    """Analyze toss impact on match results"""
    # Determine batting first team
    df['bat_first'] = df.apply(lambda x: x['toss_winner']
                            if x['toss_decision'] == 'bat'
                            else (x['team1'] if x['toss_winner'] != x['team1'] else x['team2']), axis=1)
    
    # Check if batting first team won
    df['bat_first_won'] = (df['winner'] == df['bat_first']).astype(int)
    
    # Aggregation by venue
    toss_impact = df.groupby('venue')['bat_first_won'].agg(['count', 'sum']).reset_index()
    toss_impact.columns = ['venue', 'total_matches', 'bat_first_wins']
    toss_impact['bat_first_win_pct'] = toss_impact['bat_first_wins'] / toss_impact['total_matches'] * 100
    toss_impact['field_first_win_pct'] = 100 - toss_impact['bat_first_win_pct']
    
    return toss_impact

def get_top_performers_at_venue(df, venue, min_runs=100):
    """Get top performers at a specific venue"""
    player_venue = df.groupby(['venue', 'batter']).agg(
        runs=('batsman_runs', 'sum'),
        balls=('ball', 'count'),
        dismissals=('is_wicket', 'sum'),
        matches=('match_id', 'nunique')
    ).reset_index()
    
    player_venue['strike_rate'] = (player_venue['runs'] / player_venue['balls']) * 100
    player_venue['average'] = player_venue['runs'] / player_venue['dismissals'].replace(0, 1)
    player_venue['runs_per_match'] = player_venue['runs'] / player_venue['matches']
    
    # Filter for venue and minimum runs threshold
    venue_performers = player_venue[(player_venue['venue'] == venue) & 
                                    (player_venue['runs'] >= min_runs)]
    
    # Sort by strike rate for batting performance
    top_batters = venue_performers.sort_values(by='strike_rate', ascending=False)
    
    return top_batters

def predict_player_performance(df, venue, opposition):
    """Predict player performance at a venue against specific opposition"""
    batting_df = df[df['batsman_runs'].notna() & df['batter'].notna()]
    
    # Venue-level stats
    venue_stats = (
        batting_df
        .groupby(['batter', 'venue'])
        .agg({
            'batsman_runs': 'sum',
            'ball': 'count',
            'match_id': pd.Series.nunique,
            'season': pd.Series.nunique,
        })
        .reset_index()
    )
    
    venue_stats.columns = ['batter', 'venue', 'total_runs', 'balls_faced', 'matches_played', 'seasons_played']
    venue_stats['avg_runs'] = venue_stats['total_runs'] / venue_stats['matches_played']
    venue_stats['strike_rate'] = (venue_stats['total_runs'] / venue_stats['balls_faced']) * 100
    
    # Recent form (last 5 innings)
    recent_form_df = (
        batting_df
        .groupby(['match_id', 'batter'])['batsman_runs']
        .sum()
        .reset_index()
        .sort_values(['batter', 'match_id'])
    )
    
    recent_form_df['match_order'] = recent_form_df.groupby('batter').cumcount(ascending=False)
    recent_5_avg = (
        recent_form_df[recent_form_df['match_order'] < 5]
        .groupby('batter')['batsman_runs']
        .mean()
        .reset_index()
        .rename(columns={'batsman_runs': 'recent_avg'})
    )
    
    # Opposition matchup analysis
    vs_team_avg = (
        batting_df
        .groupby(['batter', 'bowling_team'])
        .agg({'batsman_runs': 'sum', 'match_id': pd.Series.nunique})
        .reset_index()
    )
    vs_team_avg['vs_team_avg'] = vs_team_avg['batsman_runs'] / vs_team_avg['match_id']
    vs_team_avg = vs_team_avg[['batter', 'bowling_team', 'vs_team_avg']]
    
    # Filter for current venue & opponent
    merged = venue_stats[venue_stats['venue'] == venue].copy()
    merged = merged.merge(recent_5_avg, on='batter', how='left')
    
    matchup = vs_team_avg[vs_team_avg['bowling_team'] == opposition]
    merged = merged.merge(matchup, on='batter', how='left')
    
    # Fill NA values and normalize
    merged.fillna(0, inplace=True)
    
    merged['norm_avg_runs'] = merged['avg_runs'] / merged['avg_runs'].max() if merged['avg_runs'].max() > 0 else 0
    merged['norm_recent_avg'] = merged['recent_avg'] / merged['recent_avg'].max() if merged['recent_avg'].max() > 0 else 0
    merged['norm_vs_team_avg'] = merged['vs_team_avg'] / merged['vs_team_avg'].max() if merged['vs_team_avg'].max() > 0 else 0
    
    # Weighted scoring logic
    merged['predicted_score'] = (
        0.4 * merged['norm_avg_runs'] +
        0.3 * merged['norm_recent_avg'] +
        0.3 * merged['norm_vs_team_avg']
    )
    
    # Final output sorted by predicted performance
    top_batters = merged.sort_values('predicted_score', ascending=False)[
        ['batter', 'avg_runs', 'recent_avg', 'vs_team_avg', 'predicted_score', 'strike_rate']
    ]
    
    return top_batters

def predict_first_innings_score(venue_profiles, stadium_reports, selected_venue):
    """Predict first innings score based on venue characteristics"""
    # Combine venue data
    venue_data = pd.merge(venue_profiles, stadium_reports, on='venue', how='inner')
    
    # Get venue features
    venue_features = venue_data[venue_data['venue'] == selected_venue].iloc[0]
    
    # Simple prediction based on historical averages with adjustments
    base_score = venue_features['Avg 1st Innings Score']
    
    # Adjust for pitch type
    pitch_adjustments = {
        'Batting': 10,
        'Balanced': 0,
        'Bowling': -10,
        'Spin': -5,
        'Pace': -5
    }
    
    # Adjust for other factors
    bounce_adjustments = {
        'Low': -5,
        'Medium': 0,
        'High': 5
    }
    
    # Calculate adjustments
    pitch_adj = pitch_adjustments.get(venue_features['Pitch Type'], 0)
    bounce_adj = bounce_adjustments.get(venue_features['Bounce Level'], 0)
    
    # Final prediction with randomness for realism
    predicted_score = base_score + pitch_adj + bounce_adj + np.random.normal(-5, 5)
    
    # Range for prediction
    lower_bound = int(max(120, predicted_score - 10))
    upper_bound = int(predicted_score + 10)
    
    return lower_bound, upper_bound

def recommend_strategy(venue_profiles, stadium_reports, selected_venue):
    """Recommend match strategy based on venue characteristics"""
    # Combine venue data
    venue_data = pd.merge(venue_profiles, stadium_reports, on='venue', how='inner')
    
    # Get venue features
    venue_features = venue_data[venue_data['venue'] == selected_venue].iloc[0]
    
    # Decision based on historical batting first vs chasing win percentages
    bat_first_win = venue_features['Batting 1st Win %']
    chase_win = venue_features['Chasing Win %']
    
    # Consider dew factor
    dew_impact = venue_features['Dew Impact']
    
    recommendation = ""
    confidence = 0
    
    # Decision logic
    if bat_first_win > chase_win + 10:
        recommendation = "Bat First"
        confidence = min(100, (bat_first_win - chase_win) * 2)
    elif chase_win > bat_first_win + 10:
        recommendation = "Bowl First"
        confidence = min(100, (chase_win - bat_first_win) * 2)
    else:
        if dew_impact == 'High':
            recommendation = "Bowl First"
            confidence = 70
        elif dew_impact == 'Low':
            recommendation = "Bat First"
            confidence = 60
        else:
            recommendation = "Balanced - Consider Team Strengths"
            confidence = 50
    
    # Additional insights
    insights = []
    
    if venue_features['Pitch Type'] == 'Batting':
        insights.append("🏏 Batting-friendly pitch - high scoring likely")
    elif venue_features['Pitch Type'] == 'Bowling':
        insights.append("🎯 Bowling-friendly pitch - could be low scoring")
    
    if venue_features['Spin Assistance'] == 'High':
        insights.append("🔄 High spin assistance - consider extra spinners")
    
    if venue_features['Seam Movement'] == 'High':
        insights.append("⚡ Good seam movement - pace bowlers should be effective")
    
    if venue_features['Dew Impact'] == 'High':
        insights.append("💧 Heavy dew factor - defending totals can be challenging")
    
    return recommendation, confidence, insights

def identify_similar_venues(venue_profiles, stadium_reports, selected_venue):
    """Identify venues with similar playing characteristics"""
    # Combine venue data
    venue_data = pd.merge(venue_profiles, stadium_reports, on='venue', how='inner')
    
    # Features for comparison
    compare_features = ['Pitch Type', 'Bounce Level', 'Spin Assistance', 'Seam Movement',
                        'Avg 1st Innings Score', 'Avg 2nd Innings Score']
    
    # Get reference venue data
    reference = venue_data[venue_data['venue'] == selected_venue][compare_features].iloc[0]
    
    # Manually calculate similarity scores (simplified)
    similarities = []
    
    for _, row in venue_data.iterrows():
        if row['venue'] == selected_venue:
            continue
            
        # Categorical feature matching
        cat_match = sum(1 for f in ['Pitch Type', 'Bounce Level', 'Spin Assistance', 'Seam Movement'] 
                        if reference[f] == row[f])
        
        # Numerical feature similarity
        score_diff = abs(reference['Avg 1st Innings Score'] - row['Avg 1st Innings Score']) / 50
        
        # Combined similarity score (0-100)
        similarity = min(100, (cat_match / 4 * 70) + max(0, 30 - score_diff))
        
        similarities.append({
            'venue': row['venue'],
            'similarity': similarity,
            'pitch_type': row['Pitch Type'],
            'avg_score': row['Avg 1st Innings Score']
        })
    
    # Sort by similarity
    similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
    
    return similarities[:3]  # Return top 3 similar venues

def main():
    """Main function to run the Streamlit app"""
    # Load data
    df, deliveries, matches, venue_profiles, stadium_reports = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check your data files and connection.")
        return
    
    # App header
    st.markdown('<h1 class="main-header">🏏 Cricket Venue Intelligence Hub</h1>', unsafe_allow_html=True)
    
    # Create sidebar
    st.sidebar.image("https://raw.githubusercontent.com/yourusername/cricket-venue-hub/main/ipl_logo.png", 
                    use_column_width=True)
    
    # Venue selection
    venues = sorted(df['venue'].unique())
    selected_venue = st.sidebar.selectbox("Select Venue", venues)
    
    # Team selection for predictions
    teams = sorted(df['batting_team'].unique())
    team1 = st.sidebar.selectbox("Team 1", teams, index=0)
    team2 = st.sidebar.selectbox("Team 2", teams, index=1)
    
    # Process data
    venue_summary, df_filtered = create_venue_summary(df)
    phase_summary, phase_pivot = create_phase_summary(df_filtered)
    toss_impact = analyze_toss_impact(df_filtered)
    
    # Get venue-specific data
    venue_data = venue_summary[venue_summary['venue'] == selected_venue].iloc[0]
    venue_phase_data = phase_summary[phase_summary['venue'] == selected_venue]
    venue_toss_data = toss_impact[toss_impact['venue'] == selected_venue].iloc[0] if not toss_impact[toss_impact['venue'] == selected_venue].empty else None
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Venue Profile", 
        "🧠 Strategic Insights", 
        "🏅 Top Performers", 
        "🔮 Predictions",
        "🧩 Similar Venues"
    ])
    
    # Tab 1: Venue Profile
    with tab1:
        # Venue header
        st.markdown(f'<h2 class="sub-header">{selected_venue} | Venue Profile</h2>', unsafe_allow_html=True)
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric("Matches Played", f"{venue_data['total_matches']:.0f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric("Avg. 1st Innings Score", f"{venue_data['avg_1st_innings_score']:.1f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric("Avg. 2nd Innings Score", f"{venue_data['avg_2nd_innings_score']:.1f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric("Runs Per Over", f"{venue_data['runs_per_over']:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Venue characteristics
        venue_profile = venue_profiles[venue_profiles['venue'] == selected_venue].iloc[0] if not venue_profiles[venue_profiles['venue'] == selected_venue].empty else None
        stadium_report = stadium_reports[stadium_reports['venue'] == selected_venue].iloc[0] if not stadium_reports[stadium_reports['venue'] == selected_venue].empty else None
        
        if venue_profile is not None and stadium_report is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.markdown(f"<h3>Pitch Characteristics</h3>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Pitch Type:</strong> {stadium_report['Pitch Type']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Bounce Level:</strong> {stadium_report['Bounce Level']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Spin Assistance:</strong> {stadium_report['Spin Assistance']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Seam Movement:</strong> {stadium_report['Seam Movement']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Dew Impact:</strong> {stadium_report['Dew Impact']}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.markdown(f"<h3>Stadium Details</h3>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Ground Shape:</strong> {stadium_report['Ground Shape']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Boundary Dimensions:</strong> {stadium_report['Boundary Dimensions']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Altitude/Geography:</strong> {stadium_report['Altitude/Geography']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Weather Condition:</strong> {stadium_report['Weather Condition']}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Phase-wise analysis
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Phase-wise Analysis</h3>', unsafe_allow_html=True)
        
        if not venue_phase_data.empty:
            # Convert to wide format for visualization
            phase_data_wide = venue_phase_data.pivot(index='venue', columns='phase', values='run_rate').reset_index()
            
            # Create a radar chart using Plotly
            categories = ['Powerplay', 'Middle Overs', 'Death Overs']
            values = [phase_data_wide.get(cat, [0])[0] if cat in phase_data_wide.columns else 0 for cat in categories]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                line=dict(color='#3867d6'),
                fillcolor='rgba(56, 103, 214, 0.5)',
                name=selected_venue
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 12]
                    )
                ),
                title={
                    'text': f"Run Rate by Phase at {selected_venue}",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Strategic Insights
    with tab2:
        st.markdown(f'<h2 class="sub-header">{selected_venue} | Strategic Insights</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Toss decisions and outcomes
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.markdown("<h3>Toss & Match Outcomes</h3>", unsafe_allow_html=True)
            
            if venue_toss_data is not None:
                # Create bat first vs field first win percentage chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=['Bat First', 'Field First'],
                    y=[venue_toss_data['bat_first_win_pct'], venue_toss_data['field_first_win_pct']],
                    marker_color=['#3867d6', '#eb2f06'],
                    text=[f"{venue_toss_data['bat_first_win_pct']:.1f}%", 
                          f"{venue_toss_data['field_first_win_pct']:.1f}%"],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Win % by Toss Decision",
                    xaxis_title="Toss Decision",
                    yaxis_title="Win Percentage",
                    yaxis=dict(range=[0, 100]),
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                recommended_strategy, confidence, insights = recommend_strategy(
                    venue_profiles, stadium_reports, selected_venue
                )
                
                st.markdown(f"<h4>Recommended Strategy: <span class='highlight'>{recommended_strategy}</span> (Confidence: {confidence:.0f}%)</h4>", 
                          unsafe_allow_html=True)
                
                for insight in insights:
                    st.markdown(f"• {insight}")
            else:
                st.write("Insufficient data for this venue.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Scoring patterns
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.markdown("<h3>Scoring Pattern</h3>", unsafe_allow_html=True)
            
            # Filter data for the venue
            venue_data = df[df['venue'] == selected_venue]
            
            # Calculate runs per over
            if not venue_data.empty:
                runs_by_over = venue_data.groupby('over')['total_runs'].mean().reset_index()
                
                fig = px.line(runs_by_over, x='over', y='total_runs', 
                             title=f"Average Runs per Over at {selected_venue}")
                
                fig.update_traces(line=dict(color='#3867d6', width=3))
                fig.update_layout(
                    xaxis_title="Over",
                    yaxis_title="Average Runs",
                    xaxis=dict(tickmode='linear'),
                    template="plotly_white"
                )
                
                # Add vertical lines for phase divisions
                fig.add_shape(
                    type="line", x0=6, y0=0, x1=6, y1=runs_by_over['total_runs'].max()*1.1,
                    line=dict(color="red", width=2, dash="dash")
                )
                
                fig.add_shape(
                    type="line", x0=15, y0=0, x1=15, y1=runs_by_over['total_runs'].max()*1.1,
                    line=dict(color="red", width=2, dash="dash")
                )
                
                # Add annotations for phases
                fig.add_annotation(x=3, y=runs_by_over['total_runs'].max()*1.1, 
                                  text="Powerplay", showarrow=False)
                
                fig.add_annotation(x=10, y=runs_by_over['total_runs'].max()*1.1, 
                                  text="Middle Overs", showarrow=False)
                
                fig.add_annotation(x=17, y=runs_by_over['total_runs'].max()*1.1, 
                                  text="Death Overs", showarrow=False)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Insufficient data for this venue.")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 3: Top Performers
    with tab3:
        st.markdown(f'<h2 class="sub-header">{selected_venue} | Top Performers</h2>', unsafe_allow_html=True)
        
        # Get top performers
        top_batters = get_top_performers_at_venue(df, selected_venue, min_runs=100)
        
        if not top_batters.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.markdown("<h3>Batting Stars</h3>", unsafe_allow_html=True)
                
                # Display top 10 batters in a table
                display_columns = ['batter', 'runs', 'strike_rate', 'average', 'matches']
                st.dataframe(
                    top_batters[display_columns].head(10).reset_index(drop=True),
                    column_config={
                        'batter': 'Player',
                        'runs': 'Total Runs',
                        'strike_rate': st.column_config.NumberColumn('Strike Rate', format="%.2f"),
                        'average': st.column_config.NumberColumn('Average', format="%.2f"),
                        'matches': 'Matches'
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.markdown("<h3>Top Run-Scorers</h3>", unsafe_allow_html=True)
                
                # Create a bar chart of top 5 run-scorers
                top_5_runs = top_batters.sort_values('runs', ascending=False).head(5)
                
                fig = px.bar(
                    top_5_runs,
                    x='batter',
                    y='runs',
                    text='runs',
                    color='strike_rate',
                    color_continuous_scale='Viridis',
                    title=f"Top 5 Run Scorers at {selected_venue}"
                )
                
                fig.update_layout(
                    xaxis_title="Player",
                    yaxis_title="Runs",
                    xaxis={'categoryorder':'total descending'},
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Display Strike Rate vs Average scatterplot
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.markdown("<h3>Batting Performance Matrix</h3>", unsafe_allow_html=True)
            
            # Filter for minimum balls faced
            perf_data = top_batters[top_batters['balls'] >= 50].copy()
            
            fig = px.scatter(
                perf_data,
                x='strike_rate',
                y='average',
                size='runs',
                color='matches',
                hover_name='batter',
                log_x=False,
                size_max=25,
                title=f"Strike Rate vs Average at {selected_venue}"
            )
            
            # Add quadrant lines
            avg_sr = perf_data['strike_rate'].median()
            avg_avg = perf_data['average'].median()
            
            fig.add_shape(
                type="line", x0=avg_sr, y0=0, x1=avg_sr, y1=perf_data['average'].max()*1.1,
                line=dict(color="gray", width=1, dash="dash")
            )
            
            fig.add_shape(
                type="line", x0=0, y0=avg_avg, x1=perf_data['strike_rate'].max()*1.1, y1=avg_avg,
                line=dict(color="gray", width=1, dash="dash")
            )
            
            # Add quadrant labels
            fig.add_annotation(x=avg_sr/2, y=perf_data['average'].max()*0.9, 
                              text="Anchors", showarrow=False)
            
            fig.add_annotation(x=perf_data['strike_rate'].max()*0.9, y=avg_avg/2, 
                              text="Aggressive but Inconsistent", showarrow=False)
            
            fig.add_annotation(x=avg_sr/2, y=avg_avg/2, 
                              text="Struggle Zone", showarrow=False)
            
            fig.add_annotation(x=perf_data['strike_rate'].max()*0.9, y=perf_data['average'].max()*0.9, 
                              text="Match Winners", showarrow=False)
            
            fig.update_layout(
                xaxis_title="Strike Rate",
                yaxis_title="Average",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Not enough data available for performer analysis at this venue.")
    
    # Tab 4: Predictions
    with tab4:
        st.markdown(f'<h2 class="sub-header">{selected_venue} | Match Predictions</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.markdown("<h3>First Innings Score Prediction</h3>", unsafe_allow_html=True)
            
            # Predict first innings score
            lower_bound, upper_bound = predict_first_innings_score(
                venue_profiles, stadium_reports, selected_venue
            )
            
            st.markdown(f"<h2 style='text-align: center; color: #3867d6;'>{lower_bound} - {upper_bound}</h2>", 
                      unsafe_allow_html=True)
            
            st.markdown("<p style='text-align: center;'>Predicted First Innings Score Range</p>", 
                      unsafe_allow_html=True)
            
            # Add visual meter
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = (lower_bound + upper_bound) / 2,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Score Predictor"},
                gauge = {
                    'axis': {'range': [100, 220]},
                    'bar': {'color': "#3867d6"},
                    'steps': [
                        {'range': [100, 150], 'color': "lightgray"},
                        {'range': [150, 180], 'color': "gray"},
                        {'range': [180, 220], 'color': "lightblue"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 200
                    }
                }
            ))
            
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<p><i>Prediction based on venue characteristics, historical scores, and pitch conditions</i></p>", 
                      unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.markdown("<h3>Player Performance Predictions</h3>", unsafe_allow_html=True)
            
            # User selects team batting first
            batting_team = st.selectbox("Select Batting Team", [team1, team2])
            bowling_team = team2 if batting_team == team1 else team1
            
            st.markdown(f"<p>Predicting performance for <b>{batting_team}</b> batters against <b>{bowling_team}</b> at {selected_venue}</p>", 
                      unsafe_allow_html=True)
            
            # Get player predictions
            predicted_performers = predict_player_performance(df, selected_venue, bowling_team)
            
            if not predicted_performers.empty:
                # Sort and display top 5 performers
                top_5_predictions = predicted_performers.head(5)
                
                # Create horizontal bar chart colored by confidence
                fig = px.bar(
                    top_5_predictions,
                    y='batter',
                    x='predicted_score',
                    orientation='h',
                    color='predicted_score',
                    color_continuous_scale='Viridis',
                    title="Top 5 Predicted Performers",
                    text=top_5_predictions['predicted_score'].apply(lambda x: f"{x:.2f}")
                )
                
                fig.update_layout(
                    yaxis_title="",
                    xaxis_title="Performance Score",
                    yaxis={'categoryorder':'total ascending'},
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add additional context
                st.markdown("<p><b>Performance factors considered:</b></p>", unsafe_allow_html=True)
                st.markdown("• Historical venue performance (40%)")
                st.markdown("• Recent form - last 5 innings (30%)")
                st.markdown("• Record against opposition (30%)")
            else:
                st.info("Not enough data to make player performance predictions for this matchup.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Key matchups section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown("<h3>Key Venue Insights for Team Strategy</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<h4 style='text-align: center;'>Powerplay Strategy</h4>", unsafe_allow_html=True)
            
            # Get powerplay data for this venue
            powerplay_data = venue_phase_data[venue_phase_data['phase'] == 'Powerplay'] if not venue_phase_data.empty else None
            
            if powerplay_data is not None and not powerplay_data.empty:
                pp_run_rate = powerplay_data['run_rate'].values[0]
                
                st.metric("Avg Powerplay Run Rate", f"{pp_run_rate:.2f}")
                
                if pp_run_rate > 8.5:
                    st.markdown("• Aggressive approach recommended")
                    st.markdown("• Target field restrictions")
                elif pp_run_rate < 7.5:
                    st.markdown("• Conservative approach may be needed")
                    st.markdown("• Focus on preserving wickets")
                else:
                    st.markdown("• Balanced approach")
                    st.markdown("• Assess conditions early")
            else:
                st.info("No powerplay data available")
        
        with col2:
            st.markdown("<h4 style='text-align: center;'>Middle Overs</h4>", unsafe_allow_html=True)
            
            # Get middle overs data
            middle_data = venue_phase_data[venue_phase_data['phase'] == 'Middle Overs'] if not venue_phase_data.empty else None
            
            if middle_data is not None and not middle_data.empty:
                middle_run_rate = middle_data['run_rate'].values[0]
                
                st.metric("Avg Middle Overs Run Rate", f"{middle_run_rate:.2f}")
                
                if middle_run_rate > 8.5:
                    st.markdown("• Continue attacking")
                    st.markdown("• Spin may struggle here")
                elif middle_run_rate < 7.5:
                    st.markdown("• Rotate strike effectively")
                    st.markdown("• Spin likely to dominate")
                else:
                    st.markdown("• Build partnerships")
                    st.markdown("• Target boundary options")
            else:
                st.info("No middle overs data available")
        
        with col3:
            st.markdown("<h4 style='text-align: center;'>Death Overs</h4>", unsafe_allow_html=True)
            
            # Get death overs data
            death_data = venue_phase_data[venue_phase_data['phase'] == 'Death Overs'] if not venue_phase_data.empty else None
            
            if death_data is not None and not death_data.empty:
                death_run_rate = death_data['run_rate'].values[0]
                
                st.metric("Avg Death Overs Run Rate", f"{death_run_rate:.2f}")
                
                if death_run_rate > 11:
                    st.markdown("• Very high scoring at death")
                    st.markdown("• Set batters must capitalize")
                elif death_run_rate < 9:
                    st.markdown("• Difficult to accelerate")
                    st.markdown("• Plan ahead for late assault")
                else:
                    st.markdown("• Standard death over tactics")
                    st.markdown("• Focus on execution")
            else:
                st.info("No death overs data available")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 5: Similar Venues
    with tab5:
        st.markdown(f'<h2 class="sub-header">{selected_venue} | Similar Venues Analysis</h2>', unsafe_allow_html=True)
        
        # Get similar venues
        similar_venues = identify_similar_venues(venue_profiles, stadium_reports, selected_venue)
        
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown("<h3>Venues with Similar Playing Characteristics</h3>", unsafe_allow_html=True)
        st.markdown("<p>Teams that have performed well at these venues may have strategies that translate well to this venue.</p>", 
                  unsafe_allow_html=True)
        
        # Display similar venues
        for i, venue in enumerate(similar_venues):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown(f"<h4>{i+1}. {venue['venue']}</h4>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Similarity Score:</b> {venue['similarity']:.1f}%</p>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"<p><b>Pitch Type:</b> {venue['pitch_type']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Avg 1st Innings Score:</b> {venue['avg_score']:.1f}</p>", unsafe_allow_html=True)
                
                # Get teams that performed well at similar venues
                similar_venue_data = df[df['venue'] == venue['venue']]
                
                if not similar_venue_data.empty:
                    team_perf = similar_venue_data.groupby('batting_team').agg({
                        'match_id': 'nunique',
                        'total_runs': 'sum'
                    }).reset_index()
                    
                    team_perf['runs_per_match'] = team_perf['total_runs'] / team_perf['match_id']
                    top_team = team_perf.sort_values('runs_per_match', ascending=False).iloc[0]
                    
                    st.markdown(f"<p><b>Top Performing Team:</b> {top_team['batting_team']} ({top_team['runs_per_match']:.1f} runs/match)</p>", 
                              unsafe_allow_html=True)
            
            st.markdown("<hr>", unsafe_allow_html=True)
        
        # Show performance comparison across similar venues
        st.markdown("<h3>Performance Comparison Across Similar Venues</h3>", unsafe_allow_html=True)
        
        # Create list of venues for comparison
        comparison_venues = [selected_venue] + [v['venue'] for v in similar_venues]
        
        # Get phase-wise data for these venues
        comparison_data = phase_summary[phase_summary['venue'].isin(comparison_venues)]
        
        if not comparison_data.empty:
            # Create a grouped bar chart for run rates
            phase_comp = comparison_data.pivot(index='venue', columns='phase', values='run_rate').reset_index()
            
            # Melt for easier plotting
            phase_comp_melt = pd.melt(
                phase_comp, 
                id_vars=['venue'], 
                value_vars=['Powerplay', 'Middle Overs', 'Death Overs'],
                var_name='Phase', 
                value_name='Run Rate'
            )
            
            fig = px.bar(
                phase_comp_melt, 
                x='venue', 
                y='Run Rate', 
                color='Phase',
                barmode='group',
                title="Run Rate Comparison by Phase",
                color_discrete_map={
                    'Powerplay': '#3498db',
                    'Middle Overs': '#2ecc71',
                    'Death Overs': '#e74c3c'
                }
            )
            
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Run Rate",
                legend_title="Phase",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for comparison visualization.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Cricket Venue Intelligence Hub | Developed by Data Science Team</p>", 
              unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Data Last Updated: May 2025</p>", 
              unsafe_allow_html=True)

if __name__ == "__main__":
    main()
