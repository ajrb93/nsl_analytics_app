import streamlit as st
import os

# Page config
st.set_page_config(page_title="NSL Projections", layout="wide")

# Title
st.title("NSL Analytics")

# Get list of teams from the schedule folder (or hardcode them)
team_folders = ['schedule', 'performance', 'lineups']
teams = []
for folder in team_folders:
    if os.path.exists(f'images/{folder}'):
        team_files = [f.replace('.png', '') for f in os.listdir(f'images/{folder}') if f.endswith('.png')]
        teams.extend(team_files)
teams = sorted(list(set(teams)))  # Get unique team names

# Sidebar for navigation
st.sidebar.header("View Selection")
view = st.sidebar.radio(
    "Select View:",
    ["Standings", "Performance (League)", "Schedule (League)", "Schedule (Team)", "Performance (Team)", "Lineups (Team)"]
)

# Standings time period filter (only show for Standings view)
standings_period = None
if view == "Standings":
    st.sidebar.header("Time Period")
    standings_period = st.sidebar.selectbox(
        "Select Period:",
        ["Weekly", "Monthly", "Yearly"]
    )

# Team filter (only show for team-specific views)
selected_team = None
if view in ["Schedule (Team)", "Performance (Team)", "Lineups (Team)"]:
    st.sidebar.header("Team Filter")
    selected_team = st.sidebar.selectbox("Select Team:", teams)

# Display images based on selection
st.header(view)

if view == "Standings":
    # Map the selection to the file suffix
    period_map = {
        "Weekly": "w",
        "Monthly": "m",
        "Yearly": "y"
    }
    image_path = f'images/standings_{period_map[standings_period]}.png'
    
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.error(f"Image not found: {image_path}")

elif view == "Performance (League)":
    st.image('images/performance_league.png', use_container_width=True)

elif view == "Schedule (League)":
    st.image('images/schedule_league.png', use_container_width=True)

elif view == "Schedule (Team)":
    image_path = f'images/schedule/{selected_team}.png'
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.error(f"Image not found for {selected_team}")

elif view == "Performance (Team)":
    image_path = f'images/performance/{selected_team}.png'
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.error(f"Image not found for {selected_team}")

elif view == "Lineups (Team)":
    image_path = f'images/lineups/{selected_team}.png'
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.error(f"Image not found for {selected_team}")

# Optional: Add a footer with last update date
st.sidebar.markdown("---")
st.sidebar.caption("Last updated: [Add your update date here]")