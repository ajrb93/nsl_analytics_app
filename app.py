import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import base64
from io import BytesIO
import numpy as np

# --- 1. CONFIG & COMPACT STYLING ---
st.set_page_config(layout="wide", page_title="Northern Super League")

# CUSTOM CSS: Shrinks headers, table padding, and overall container gaps
st.markdown("""
    <style>
    /* Page margins */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* Tab labels */
    button[data-baseweb="tab"] {
        font-size: 14px !important;
    }
    button[data-baseweb="tab"] div {
        font-size: 14px !important;
    }

    /* Expander headers */
    div[data-testid="stExpander"] div[role="button"] p { 
        font-size: 12px !important; 
        font-weight: bold !important; 
    }
    </style>
""", unsafe_allow_html=True)

# Load All Data
def credible_range_str(row, level=0.9):
    probs = row.sort_values(ascending=False)
    cumsum = probs.cumsum()
    included = probs.index[cumsum <= level]
    if len(included) < len(probs):
        included = included.append(pd.Index([cumsum.index[len(included)]]))
    nums = [int(float(p)) for p in included]
    lo, hi = min(nums), max(nums)
    return f"{lo}" if lo == hi else f"{lo} to {hi}"

def load_standings_sims():
    files = os.listdir('data/Sim_States/')
    files = list(filter(lambda k: '.ftr' in k, files))

    standings_files = list(filter(lambda k: '_matches' not in k, files))
    match_files = list(filter(lambda k: '_matches' in k, files))

    standings_sims = []
    for file in standings_files:
        temp = pd.read_feather('data/Sim_States/'+file)
        date = file.replace('.ftr','')
        temp['Sim_Date'] = date
        standings_sims.append(temp)
    standings_sims = pd.concat((standings_sims)).reset_index()
    standings_sims.Sim_Date = pd.to_datetime(standings_sims.Sim_Date).dt.date
    standings_sims = standings_sims.fillna(0)
    standings_sims['Champ'] = standings_sims['1']
    standings_sims['Playoffs'] = standings_sims[['1','2','3','4']].sum(axis=1)
    standings_sims['Last'] = standings_sims[['6']].sum(axis=1)

    standings_sims['range'] = standings_sims[['1','2','3','4','5','6']].apply(credible_range_str, axis=1)
    standings_sims['season'] = pd.to_datetime(standings_sims.Sim_Date).dt.year
    match_sims = []
    for file in match_files:
        temp = pd.read_feather('data/Sim_States/'+file)
        match_sims.append(temp)
    match_sims = pd.concat((match_sims))

    return standings_sims, match_sims

def create_standings_file(standings,standings_sims,team_ratings,season,max_date,min_date):
    temp = standings[standings.season == season][['season','F_team','F_score','A_score','F_p','F_xg','A_xg','F_xpts','oRTG','dRTG','nRTG']].reset_index(drop=True)
    temp['GD'] = temp.F_score - temp.A_score
    temp['xGD'] = temp.F_xg - temp.A_xg
    temp_sim = standings_sims[standings_sims.Sim_Date == max_date].set_index('index')[['Points','Champ','Playoffs','Last','range']]
    temp_sim2 = standings_sims[standings_sims.Sim_Date == min_date].set_index('index')[['Points','Champ','Playoffs','Last']]
    temp_sim2 = temp_sim - temp_sim2
    temp_sim3 = team_ratings[team_ratings.Date == max_date].set_index('Team')[['Date','A','B','C']]
    temp_sim4 = team_ratings[team_ratings.Date == min_date].set_index('Team')[['A','B','C']]
    temp_sim4 = temp_sim3 - temp_sim4
    temp = temp.merge(temp_sim.reset_index(),left_on='F_team',right_on='index').merge(temp_sim2.reset_index(),left_on='F_team',right_on='index',suffixes=['','_c']).merge(
        temp_sim3.reset_index(),left_on='F_team',right_on='Team').merge(temp_sim4.reset_index(),left_on='F_team',right_on='Team',suffixes=['','_c'])
    temp = temp[['season','Team','C','C_c','A','A_c','B','B_c','nRTG','oRTG','dRTG','Points','Points_c','F_p','F_xpts','GD','xGD','Champ','Champ_c','Playoffs','Playoffs_c','Last',
                 'Last_c','range']].rename(
                     columns={'A':'oPRE','A_c':'oPREΔ','B':'dPRE','B_c':'dPREΔ','F_p':'P','F_xpts':'xpts','Points':'Proj','Points_c':'ProjΔ','C':'nPRE','C_c':'nPREΔ',
                               'Champ':'Win','Champ_c':'WinΔ','Playoffs_c':'PlayoffsΔ','Last_c':'LastΔ'})
    return temp

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3],16) for i in range(0,lv,lv//3))

def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb

def mean_color(color1,color2):
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    
    avg = lambda x,y: round((x+y)/2)
    new_rgb = ()
    for i in range(len(rgb1)):
        new_rgb += (avg(rgb1[i],rgb2[i]),)
    
    return '#' + rgb_to_hex(new_rgb)

#colormap
colors = [(0.75,0,0),(1,1,1),(0,0.75,0)]
colors_r = [(0,0.75,0),(1,1,1),(0.75,0,0),]
n_bins = 100
cmap = mcolors.LinearSegmentedColormap.from_list('redwhitegreen',colors,N=n_bins)
cmap_r = mcolors.LinearSegmentedColormap.from_list('redwhitegreen_r',colors_r,N=n_bins)
norm_o = mcolors.TwoSlopeNorm(vmin=0,vcenter=1.35,vmax=2.6)
norm_r = mcolors.TwoSlopeNorm(vmin=0,vcenter=1,vmax=2)
norm_p = mcolors.TwoSlopeNorm(vmin=0,vcenter=1.5,vmax=3)
norm_w = mcolors.TwoSlopeNorm(vmin=0,vcenter=1/4,vmax=1)
norm_perf = mcolors.TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)

def plot_standings_table(standings_df):
    fig, ax = plt.subplots(figsize=(12,6/2.33333333))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # --- HEADERS ---
    ax.annotate('Team',       (0.01,  0.97), va='center', ha='left',   size=10, weight='bold')
    ax.annotate('Skill',      (1.65/10, 0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('Off',        (2.65/10, 0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('Def',        (3.65/10, 0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('Performance',(4.65/10, 0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('Proj',       (5.5/10,  0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('Points',     (6.15/10, 0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('GD',         (6.7/10,  0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('Champ',      (7.45/10, 0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('Playoff',    (8.3/10,  0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('Last',       (9.1/10,  0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('Range',      (9.75/10, 0.97), va='center', ha='center', size=10, weight='bold')

    # --- VERTICAL DIVIDERS ---
    for x in [1.15, 2.15, 3.15, 4.15, 5.15, 5.85, 6.45, 7.05, 7.85, 8.65, 9.50]:
        ax.axvline(x/10, color='black', linewidth=0.5)

    # --- ROWS ---
    n_teams = len(standings_df)
    top = 0.93
    bottom_margin = 0.01
    total_height = top - bottom_margin
    space = total_height / n_teams
    i_loc = top - space / 2

    ax.vlines(4.483/10, bottom_margin, top, color='black', linewidth=0.3, linestyle='--')
    ax.vlines(4.816/10, bottom_margin, top, color='black', linewidth=0.3, linestyle='--')

    for _, row in standings_df.iterrows():
        team = row['Team']

        # Team name
        ax.annotate(team, (0.01, i_loc), va='center', ha='left', size=9, fontweight='bold',color=team_colors[team]['home_secondary'])
        ax.add_patch(Rectangle((0,i_loc+space/2),1.15/10,-space,facecolor=team_colors[team]['home_primary']))
        ax.add_patch(Rectangle((1.15/10, i_loc - space/2), 1, space, facecolor = mean_color(mean_color(team_colors[team]['home_primary'],'#FFFFFF'),'#FFFFFF')))

        # Skill (nPRE)
        ax.annotate(f"{row['nPRE']:.0%}", (1.4/10, i_loc), va='center', ha='center', size=9)
        delta_color = 'darkgreen' if row['nPREΔ'] > 0 else 'darkred'
        ax.annotate(f"({'+' if row['nPREΔ'] > 0 else ''}{row['nPREΔ']:.0%})", (1.9/10, i_loc), va='center', ha='center', size=9, color=delta_color)
        ax.add_patch(Rectangle((1.15/10, i_loc - space/2), 0.5/10, space,facecolor=cmap(row['nPRE'])))

        # Offensive (oPRE)
        ax.annotate(f"{row['oPRE']:.2f}", (2.4/10, i_loc), va='center', ha='center', size=9)
        delta_color = 'darkgreen' if row['oPREΔ'] > 0 else 'darkred'
        ax.annotate(f"({'+' if row['oPREΔ'] > 0 else ''}{row['oPREΔ']:.0%})", (2.9/10, i_loc), va='center', ha='center', size=9, color=delta_color)
        ax.add_patch(Rectangle((2.15/10, i_loc - space/2), 0.5/10, space,facecolor=cmap(norm_r(row['oPRE']))))

        # Defensive (dPRE)
        ax.annotate(f"{row['dPRE']:.2f}", (3.4/10, i_loc), va='center', ha='center', size=9)
        delta_color = 'darkgreen' if row['dPREΔ'] < 0 else 'darkred'
        ax.annotate(f"({'+' if row['dPREΔ'] < 0 else ''}{row['dPREΔ']*-1:.0%})", (3.9/10, i_loc), va='center', ha='center', size=9, color=delta_color)
        ax.add_patch(Rectangle((3.15/10, i_loc - space/2), 0.5/10, space,facecolor=cmap(1 - norm_r(row['dPRE']))))

        # Performance (nRTG, oRTG, dRTG)
        ax.add_patch(Rectangle((4.15/10, i_loc - space/2), (1/3)/10, space,facecolor=cmap(norm_perf(row['nRTG']))))
        ax.add_patch(Rectangle((4.483/10, i_loc - space/2), (1/3)/10, space,facecolor=cmap(norm_o(row['oRTG']))))
        ax.add_patch(Rectangle((4.816/10, i_loc - space/2), (1/3)/10, space,facecolor=cmap(1 - norm_o(row['dRTG']))))
        ax.annotate(f"{row['nRTG']:.2f}", (4.32/10, i_loc), va='center', ha='center', size=9)
        ax.annotate(f"{row['oRTG']:.2f}", (4.65/10, i_loc), va='center', ha='center', size=9)
        ax.annotate(f"{row['dRTG']:.2f}", (4.97/10, i_loc), va='center', ha='center', size=9)

        # Proj + ProjΔ
        ax.annotate(f"{row['Proj']:.0f}", (5.3/10, i_loc), va='center', ha='center', size=9)
        delta_color = 'darkgreen' if row['ProjΔ'] > 0 else 'darkred'
        ax.annotate(f"({'+' if row['ProjΔ'] > 0 else ''}{row['ProjΔ']:.0f})", (5.65/10, i_loc), va='center', ha='center', size=9, color=delta_color)

        # Points + xPts
        ax.annotate(f"{int(row['P'])}", (6.0/10, i_loc), va='center', ha='center', size=9)
        ax.annotate(f"{row['xpts']:.1f}", (6.25/10, i_loc), va='center', ha='center', size=9)

        # GD + xGD
        ax.annotate(f"{int(row['GD'])}", (6.6/10, i_loc), va='center', ha='center', size=9)
        ax.annotate(f"{row['xGD']:.1f}", (6.85/10, i_loc), va='center', ha='center', size=9)

        # Champ + WinΔ
        ax.annotate(f"{row['Win']:.0%}", (7.25/10, i_loc), va='center', ha='center', size=9)
        delta_color = 'darkgreen' if row['WinΔ'] > 0 else 'darkred'
        ax.annotate(f"({'+' if row['WinΔ'] > 0 else ''}{row['WinΔ']:.0%})", (7.65/10, i_loc), va='center', ha='center', size=9, color=delta_color)

        # CL + CLΔ
        ax.annotate(f"{row['Playoffs']:.0%}", (8.05/10, i_loc), va='center', ha='center', size=9)
        delta_color = 'darkgreen' if row['PlayoffsΔ'] > 0 else 'darkred'
        ax.annotate(f"({'+' if row['PlayoffsΔ'] > 0 else ''}{row['PlayoffsΔ']:.0%})", (8.45/10, i_loc), va='center', ha='center', size=9, color=delta_color)

        # Rel + RelΔ
        ax.annotate(f"{row['Last']:.0%}", (8.85/10, i_loc), va='center', ha='center', size=9)
        delta_color = 'darkgreen' if row['LastΔ'] < 0 else 'darkred'
        ax.annotate(f"({'+' if row['LastΔ'] < 0 else ''}{row['LastΔ']*-1:.0%})", (9.25/10, i_loc), va='center', ha='center', size=9, color=delta_color)

        # Range
        ax.annotate(row['range'], (9.75/10, i_loc), va='center', ha='center', size=9)

        # Row divider
        ax.axhline(i_loc - space/2, color='black', linewidth=0.5)

        i_loc -= space

    # Top border
    ax.axhline(0.935, color='black', linewidth=0.5)

    plt.tight_layout()
    return fig

def plot_ratings_scatter(standings_df, team_colors):
    fig = go.Figure()

    off_mean = standings_df['oPRE'].mean()
    def_mean = standings_df['dPRE'].mean()

    # Diagonal reference lines (equivalent to your matplotlib lines)
    for offset in [-2/3, -1/3, 0, 1/3, 2/3]:
        x_start = def_mean * 1.5
        x_end = def_mean * 0.5
        fig.add_trace(go.Scatter(
            x=[x_start, x_end],
            y=[x_start + offset * off_mean, x_end + offset * off_mean],
            mode='lines',
            line=dict(color='black', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Arrows showing movement
    for _, row in standings_df.iterrows():
        team = row['Team']
        primary = team_colors[team]['home_primary']
        x_start = row['dPRE'] - row['dPREΔ']
        y_start = row['oPRE'] - row['oPREΔ']
        x_end = row['dPRE']
        y_end = row['oPRE']

        # Arrow line
        fig.add_trace(go.Scatter(
            x=[x_start, x_end],
            y=[y_start, y_end],
            mode='lines',
            line=dict(color=primary, width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Scatter points at current position
    fig.add_trace(go.Scatter(
        x=standings_df['dPRE'],
        y=standings_df['oPRE'],
        mode='markers',
        marker=dict(
            color=[team_colors[t]['home_primary'] for t in standings_df['Team']],
            size=12,
            line=dict(
                color=[team_colors[t]['home_secondary'] for t in standings_df['Team']],
                width=2
            )
        ),
        text=standings_df['Team'],
        hovertemplate='<b>%{text}</b><br>Off: %{y:.2f}<br>Def: %{x:.2f}<extra></extra>',
        showlegend=False
    ))

    fig.update_layout(
        xaxis=dict(
            range=[def_mean * 1.5, def_mean * 0.5],  # inverted — lower def is better
            title='Defensive Rating',
            showgrid=False
        ),
        yaxis=dict(
            range=[off_mean * 0.5, off_mean * 1.5],
            title='Offensive Rating',
            showgrid=False,
            scaleanchor='x',
            scaleratio=1
        ),
        plot_bgcolor='gainsboro',
        margin=dict(l=20, r=20, t=20, b=20),
        height = 400,
        width = 400
    )

    return fig

def plot_position_heatmap(standings_sims, standings_df, selected_end_date, team_colors):
    # Get position probabilities for selected date, ordered by current standings
    sim_data = standings_sims[standings_sims.Sim_Date == selected_end_date].set_index('index')
    position_cols = [str(i) for i in range(1, 7)]
    
    # Order teams by current points (same order as standings table)
    teams_ordered = standings_df['Team'].tolist()
    
    heatmap_data = sim_data.loc[teams_ordered, position_cols].astype(float)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=list(range(1, 7)),
        y=teams_ordered,
        colorscale='RdYlGn',
        showscale=False,
        hovertemplate='<b>%{y}</b><br>Position %{x}: %{z:.0%}<extra></extra>',
        zmin=0,
        zmax=1,
        text=[[f"{val*100:.0f}" if val >= 0.005 else "" for val in row] for row in heatmap_data.values],
        texttemplate="%{text}",
        textfont=dict(size=9)
    ))

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(
            autorange='reversed',  # keep standings order top to bottom
            tickfont=dict(size=10)
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        width=420,  # 400px plot + room for team names on left
        height=100
    )

    return fig

def create_matches_df(match_sims,matches,team_ratings,selected_season,selected_end_date):
    match_sims = match_sims[match_sims.Sim_Date <= selected_end_date].sort_values('Sim_Date').groupby(['date','Home','Away']).tail(1)
    match_sims = match_sims[['Sim_Date','date','Home','Away','h_exp','a_exp','h_win','d_win','a_win']]
    matches = matches[matches.season == selected_season]
    matches = matches[['date','home_team','away_team','home_score','away_score','home_xg','away_xg','home_p','away_p','home_perf','away_perf','home_xpts','away_xpts']]
    matches.loc[pd.to_datetime(matches.date).dt.date > selected_end_date,('home_score','away_score','home_xg','away_xg','home_p','away_p','home_perf','away_perf',
                                                       'home_xpts','away_xpts')] = np.nan
    matches = matches.merge(match_sims,left_on=['date','home_team','away_team'],right_on=['date','Home','Away']).drop(columns=['Home','Away'])

    plot_df = matches.merge(
        team_ratings.drop(columns=['Season','A','B']), left_on=['Sim_Date','home_team'], right_on=['Date','Team']).merge(
        team_ratings.drop(columns=['Season','A','B']), left_on=['Sim_Date','away_team'], right_on=['Date','Team'], suffixes=['_H','_A']).drop(
        columns=['Sim_Date','Date_H','Date_A'])
    plot_df['Pre_Pts_H'] = plot_df.h_win * 3 + plot_df.d_win
    plot_df['Pre_Pts_A'] = plot_df.a_win * 3 + plot_df.d_win
    return plot_df.sort_values('date').reset_index(drop=True)

def create_results_figure(plot_df):
    results = plot_df[~plot_df.home_score.isna()].reset_index(drop=True).sort_values('date',ascending=False)
    results[' '] = ''
    results['score'] = results.home_score.astype('int').astype('str') + ' - ' + results.away_score.astype('int').astype('str')
    results = results[['date','home_team','Pre_Pts_H','score','Pre_Pts_A','away_team',' ','home_xpts','away_xpts',' ','home_xg','away_xg','home_score','away_score']].rename(
        columns={'date':'Date','home_team':'Home','Pre_Pts_H':'H_F','Pre_Pts_A':'A_F','away_team':'Away','home_xpts':'Per_H','away_xpts':'Per_A',
                 'home_xg':'xG_H','away_xg':'xG_A'})

    fig_height = max(2, len(results) * 0.2)
    fig, ax = plt.subplots(figsize=(7,fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Column x positions
    col_x = {
        'Date':   0.01,
        'Home':   0.12,
        '':    0.37,
        'Away':   0.42,
        '1':0.67,
        'HExp':0.68,
        'HPer':  0.73,
        '3':0.78,
        'AExp':0.79,
        'APer':  0.84,
        '2':0.89,
        'HxG':   0.90,
        'AxG':   0.95
    }

    # Headers
    header_y = (len(results)+0.5)/(len(results)+1)
    for col, x in col_x.items():
        if (col == '1') | (col == '2') | (col == '3'):
            pass
        else:
            ha = 'left'
            ax.annotate(col, (x, header_y), va='center', ha=ha, size=7, weight='bold')

    # Row layout
    top = len(results)/(len(results)+1)
    ax.axhline(top, color='black', linewidth=0.8)
    bottom_margin = 1/(len(results)+1)/10
    total_height = top - bottom_margin
    space = total_height / max(4, len(results))
    i_loc = top - space / 2

    # Vertical dividers (between groups)
    for x in col_x.values():
        ax.vlines(x-0.005, bottom_margin, top, color='black', linewidth=0.5)

    for _, row in results.iterrows():
        home_primary = team_colors[row['Home']]['home_primary']
        home_text = team_colors[row['Home']]['home_secondary']
        away_primary = team_colors[row['Away']]['home_primary']
        away_text = team_colors[row['Away']]['home_secondary']

        # Home team background
        ax.add_patch(Rectangle((0.12-0.005, i_loc - space/2), 0.25, space, facecolor=home_primary))
        # Away team background
        ax.add_patch(Rectangle((0.42-0.005, i_loc - space/2), 0.25, space, facecolor=away_primary))

        # Performance color rectangles
        ax.add_patch(Rectangle((0.68-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(norm_p(row['H_F'])) if pd.notna(row['H_F']) else 'lightgray'))
        ax.add_patch(Rectangle((0.73-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(norm_p(row['Per_H'])) if pd.notna(row['Per_H']) else 'lightgray'))
        ax.add_patch(Rectangle((0.79-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(norm_p(row['A_F'])) if pd.notna(row['A_F']) else 'lightgray'))
        ax.add_patch(Rectangle((0.84-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(norm_p(row['Per_A'])) if pd.notna(row['Per_A']) else 'lightgray'))
        ax.add_patch(Rectangle((0.9-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(norm_o(row['xG_H'])) if pd.notna(row['xG_H']) else 'lightgray'))
        ax.add_patch(Rectangle((0.95-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(norm_o(row['xG_A'])) if pd.notna(row['xG_A']) else 'lightgray'))
        
        if row['home_score'] > row['away_score']:
            primary = home_primary
            text = home_text
        elif row['home_score'] < row['away_score']:
            primary = away_primary
            text = away_text
        else:
            primary = 'white'
            text = 'black'

        ax.add_patch(Rectangle((0.37-0.005,i_loc - space/2),0.05,space,facecolor=primary))

        # Text annotations
        ax.annotate(str(row['Date'])[:10], (col_x['Date'], i_loc), va='center', ha='left', size=7)
        ax.annotate(row['Home'], (col_x['Home'], i_loc), va='center', ha='left', size=7, 
                    color=home_text, fontweight='bold')
        ax.annotate(f"{row['H_F']:.2f}", (col_x['HExp'], i_loc), va='center', ha='left', size=7)
        ax.annotate(row['score'], (col_x[''], i_loc), va='center', ha='left', size=7, fontweight='bold',color=text)
        ax.annotate(f"{row['A_F']:.2f}", (col_x['AExp'], i_loc), va='center', ha='left', size=7)
        ax.annotate(row['Away'], (col_x['Away'], i_loc), va='center', ha='left', size=7,
                    color=away_text, fontweight='bold')
        ax.annotate(f"{row['Per_H']:.2f}" if pd.notna(row['Per_H']) else '', 
                    (col_x['HPer'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['Per_A']:.2f}" if pd.notna(row['Per_A']) else '', 
                    (col_x['APer'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['xG_H']:.2f}" if pd.notna(row['xG_H']) else '', 
                    (col_x['HxG'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['xG_A']:.2f}" if pd.notna(row['xG_A']) else '', 
                    (col_x['AxG'], i_loc), va='center', ha='left', size=7)

        # Row divider
        ax.axhline(i_loc - space/2, color='black', linewidth=0.3)
        i_loc -= space
    return fig

def create_schedule_figure(plot_df):
    results = plot_df[plot_df.home_score.isna()].reset_index(drop=True).sort_values('date',ascending=True)
    results[' '] = ''
    results['score'] = ''
    results = results[['date','home_team','Pre_Pts_H','score','Pre_Pts_A','away_team',' ','C_H','C_A',' ','h_win','d_win','a_win']].rename(
        columns={'date':'Date','home_team':'Home','Pre_Pts_H':'H_F','Pre_Pts_A':'A_F','away_team':'Away','C_H':'HRtg','C_A':'ARtg',
                 'h_exp':'HpG','a_exp':'ApG'})

    fig_height = max(2, len(results) * 0.2)
    fig, ax = plt.subplots(figsize=(7,fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Column x positions
    col_x = {
        'Date':   0.01,
        'Home':   0.12,
        'H':0.37,
        'D':0.42,
        'A':0.48,
        'Away':0.53,
        '1':0.78,
        'HRtg':0.79,
        'ARtg':0.84,
        '2':0.89,
        'HExp':0.90,
        'AExp':0.95
    }

    # Headers
    header_y = (len(results)+0.5)/(len(results)+1)
    for col, x in col_x.items():
        if (col == '1') | (col == '2'):
            pass
        else:
            ha = 'left'
            shift = 0.015 if col in ['H','A','D'] else 0
            ax.annotate(col, (x + shift , header_y), va='center', ha=ha, size=7, weight='bold')

    # Row layout
    top = len(results)/(len(results)+1)
    ax.axhline(top, color='black', linewidth=0.8)
    bottom_margin = 1/(len(results)+1)/10
    total_height = top - bottom_margin
    space = total_height / max(4,len(results))
    i_loc = top - space / 2

    # Vertical dividers (between groups)
    for x in col_x.values():
        ax.vlines(x-0.005, bottom_margin, top, color='black', linewidth=0.5)

    for _, row in results.iterrows():
        home_primary = team_colors[row['Home']]['home_primary']
        home_text = team_colors[row['Home']]['home_secondary']
        away_primary = team_colors[row['Away']]['home_primary']
        away_text = team_colors[row['Away']]['home_secondary']

        # Home team background
        ax.add_patch(Rectangle((0.12-0.005, i_loc - space/2), 0.25, space, facecolor=home_primary))
        # Away team background
        ax.add_patch(Rectangle((0.53-0.005, i_loc - space/2), 0.25, space, facecolor=away_primary))

        # Performance color rectangles
        ax.add_patch(Rectangle((0.37-0.005, i_loc - space/2), 0.05, space,
                    facecolor=cmap(norm_w(row['h_win'])) if pd.notna(row['h_win']) else 'lightgray'))
        ax.add_patch(Rectangle((0.42-0.005, i_loc - space/2), 0.06, space,
                    facecolor=cmap(norm_w(row['d_win'])) if pd.notna(row['d_win']) else 'lightgray'))
        ax.add_patch(Rectangle((0.48-0.005, i_loc - space/2), 0.05, space,
                    facecolor=cmap(norm_w(row['a_win'])) if pd.notna(row['a_win']) else 'lightgray'))

        ax.add_patch(Rectangle((0.79-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(row['HRtg']) if pd.notna(row['HRtg']) else 'lightgray'))
        ax.add_patch(Rectangle((0.84-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(row['ARtg']) if pd.notna(row['ARtg']) else 'lightgray'))
        ax.add_patch(Rectangle((0.9-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(norm_p(row['H_F'])) if pd.notna(row['H_F']) else 'lightgray'))
        ax.add_patch(Rectangle((0.95-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(norm_p(row['A_F'])) if pd.notna(row['A_F']) else 'lightgray'))

        # Text annotations
        ax.annotate(str(row['Date'])[:10], (col_x['Date'], i_loc), va='center', ha='left', size=7)
        ax.annotate(row['Home'], (col_x['Home'], i_loc), va='center', ha='left', size=7, 
                    color=home_text, fontweight='bold')
        ax.annotate(f"{row['h_win']:.0%}", (col_x['H'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['d_win']:.0%}", (col_x['D']+0.005, i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['a_win']:.0%}", (col_x['A'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['H_F']:.2f}", (col_x['HExp'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['A_F']:.2f}", (col_x['AExp'], i_loc), va='center', ha='left', size=7)
        ax.annotate(row['Away'], (col_x['Away'], i_loc), va='center', ha='left', size=7,
                    color=away_text, fontweight='bold')
        ax.annotate(f"{row['HRtg']:.0%}" if pd.notna(row['HRtg']) else '', 
                    (col_x['HRtg'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['ARtg']:.0%}" if pd.notna(row['ARtg']) else '', 
                    (col_x['ARtg'], i_loc), va='center', ha='left', size=7)

        # Row divider
        ax.axhline(i_loc - space/2, color='black', linewidth=0.3)
        i_loc -= space
    return fig

def scrollable_plot(fig, height=400):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    
    st.markdown(f"""
        <div style="height:{height}px; overflow-y:scroll; overflow-x:hidden;">
            <img src="data:image/png;base64,{img_base64}" style="width:100%;">
        </div>
    """, unsafe_allow_html=True)

def create_player_mvps(player_stats,matches_df,selected_season,selected_end_date):
    player_stats = player_stats[(player_stats.season == selected_season) & (player_stats.Date.dt.date <= selected_end_date)].reset_index(drop=True)
    player_stats = player_stats[player_stats.Type == 'Regular']
    player_stats = player_stats.merge(pd.pivot_table(
        player_stats,index='Name',columns='P',values='MIN',aggfunc='sum').fillna(0).idxmax(axis=1).reset_index().rename(columns={0:'Pos'}))
    player_mvp = player_stats.groupby(['Name','Pos']).agg(
        {'MIN':'sum','Rtg':'sum','Team': lambda x: ', '.join(sorted(set(x)))})
    player_mvp['per90'] = player_mvp.Rtg / player_mvp.MIN * 90
    player_mvp = player_mvp.drop(columns=['MIN']).rename(columns={'Rtg':'Goals Added','per90':'GA per 90'}).reset_index()
    
    return player_mvp.reset_index()
    
def create_mvp_figure(plot_df):
    mvps = plot_df.sort_values('Goals Added',ascending=False)

    fig, ax = plt.subplots(figsize=(8,40))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Column x positions
    col_x = {
        'Player':   0.01,
        'Pos':0.505,
        'Goals+':0.565,
        'G+/90':   0.805}

    # Headers
    header_y = (len(mvps)+0.5)/(len(mvps)+1)
    for col, x in col_x.items():
        ha = 'left'
        ax.annotate(col, (x , header_y), va='center', ha=ha, size=7, weight='bold')

    # Row layout
    top = len(mvps)/(len(mvps)+1)
    ax.axhline(top, color='black', linewidth=0.8)
    bottom_margin = 1/(len(mvps)+1)/10
    total_height = top - bottom_margin
    space = total_height / max(4,len(mvps))
    i_loc = top - space / 2

    # Vertical dividers (between groups)
    for x in col_x.values():
        ax.vlines(x-0.005, bottom_margin, top, color='black', linewidth=0.5)

    for _, row in mvps.iterrows():
        if len(row['Team']) > 1:
            primary = 'white'
            secondary = 'black'
        else:
            primary = team_colors[row['team'][0]]['home_primary']
            secondary = team_colors[row['team'][0]]['home_secondary']

        ax.add_patch(Rectangle((0, i_loc - space/2),1, space, facecolor=primary))
        # Text annotations
        ax.annotate(row['Name'], (col_x['Player'], i_loc), va='center', ha='left', size=7,color = secondary,fontweight='bold')
        ax.annotate(row['Pos'], (col_x['Pos'], i_loc), va='center', ha='left', size=7,color = secondary)
        ax.annotate(f"{row['Goals Added']:.2f}" if pd.notna(row['Goals Added']) else '', (col_x['Goals+'], i_loc), va='center', ha='left', size=7,color = secondary)
        ax.annotate(f"{row['GA per 90']:.2f}" if pd.notna(row['GA per 90']) else '', (col_x['G+/90'], i_loc), va='center', ha='left', size=7,color = secondary)
        # Row divider
        ax.axhline(i_loc - space/2, color='black', linewidth=0.3)
        i_loc -= space
    return fig

def create_multi_year_standings(team_ratings,standings):
    temp1 = team_ratings.groupby('Team').head(1).reset_index()
    temp1.Season = (temp1.Season.astype('int') - 1)
    temp2 = team_ratings.groupby(['Team','Season']).tail(1).reset_index()
    temp_season_ratings = pd.concat((temp1,temp2))[['Season','Team','A','B','C']].sort_values(['Team','Season'])
    temp_season_ratings[['A_C','B_C','C_C']] = temp_season_ratings[['A','B','C']] - temp_season_ratings.groupby('Team')[['A','B','C']].shift(periods=1)

    temp_standings = standings.copy()
    temp_standings['GD'] = temp_standings.F_score - temp_standings.A_score
    temp_standings['xGD'] = temp_standings.F_xg - temp_standings.A_xg
    temp_standings = temp_standings.sort_values(['F_p','GD','F_score'],ascending=False)
    temp_standings['Rank'] = 1
    temp_standings.Rank = temp_standings.groupby('season').Rank.cumsum()
    temp_standings = temp_standings[['season','F_team','F_p','F_xpts','GD','xGD','Rank']].rename(
        columns={'season':'Season','F_team':'Team'})
    return temp_season_ratings.merge(temp_standings,on=['Team','Season']).sort_values('Season',ascending=False)

def plot_history_table(results):
    fig, ax = plt.subplots(figsize=(5,1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Column x positions
    col_x = {
        '1':   0.01,
        'Rk':0.09,
        'Points':   0.14,
        'GD':0.29,
        'Off':0.44,
        'Def':0.62,
        'Skill':0.80
    }

    # Headers
    header_y = (len(results)+0.5)/(len(results)+1)
    for col, x in col_x.items():
        if (col == '1') | (col == '2'):
            pass
        else:
            ha = 'left'
            
            ax.annotate(col, (x , header_y), va='center', ha=ha, size=7, weight='bold')

    # Row layout
    top = len(results)/(len(results)+0.8)
    ax.axhline(top, color='black', linewidth=0.8)
    bottom_margin = 1/(len(results)+1)/10
    total_height = top - bottom_margin
    space = total_height / len(results)
    i_loc = top - space / 2

    norm_rk = mcolors.TwoSlopeNorm(vmin=1,vcenter=3.5,vmax=6)
    

    # Vertical dividers (between groups)
    for x in col_x.values():
        ax.vlines(x-0.005, bottom_margin, top, color='black', linewidth=0.5)


    for _, row in results.iterrows():
        ax.annotate('20'+str(row['Season']), (col_x['1']+0.035, i_loc), va='center', ha='center', size=7)
        ax.annotate(f"{int(row['F_p'])}", (col_x['Points']+0.035, i_loc), va='center', ha='center', size=7)
        ax.annotate(f"{row['F_xpts']:.1f}", (col_x['Points']+0.095, i_loc), va='center', ha='center', size=7)
        ax.annotate(f"{int(row['GD'])}", (col_x['GD']+0.035, i_loc), va='center', ha='center', size=7)
        ax.annotate(f"{row['xGD']:.1f}", (col_x['GD']+0.095, i_loc), va='center', ha='center', size=7)
        ax.annotate(f"{int(row['Rank'])}", (col_x['Rk']+0.02, i_loc), va='center', ha='center', size=7)
        ax.add_patch(Rectangle((0.09-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap_r(norm_rk(row['Rank'])) if pd.notna(row['Rank']) else 'lightgray'))

        ax.annotate(f"{row['A']:.2f}", (col_x['Off']+0.035, i_loc), va='center', ha='center', size=7)
        delta_color = 'darkgreen' if row['A_C'] > 0 else 'darkred'
        ax.annotate(f"({'+' if row['A_C'] > 0 else ''}{row['A_C']:.0%})", (col_x['Off']+0.12, i_loc), va='center', ha='center', size=7, color=delta_color)
        ax.add_patch(Rectangle((col_x['Off']-0.005, i_loc - space/2), 0.075, space,facecolor=cmap(norm_r(row['A']))))

        ax.annotate(f"{row['B']:.2f}", (col_x['Def']+0.035, i_loc), va='center', ha='center', size=7)
        delta_color = 'darkgreen' if row['B_C'] < 0 else 'darkred'
        ax.annotate(f"({'+' if row['B_C'] < 0 else ''}{row['B_C']*-1:.0%})", (col_x['Def']+0.12, i_loc), va='center', ha='center', size=7, color=delta_color)
        ax.add_patch(Rectangle((col_x['Def']-0.005, i_loc - space/2), 0.075, space,facecolor=cmap_r(norm_r(row['B']))))

        ax.annotate(f"{row['C']:.0%}", (col_x['Skill']+0.035, i_loc), va='center', ha='center', size=7)
        delta_color = 'darkgreen' if row['C_C'] > 0 else 'darkred'
        ax.annotate(f"({'+' if row['C_C'] > 0 else ''}{row['C_C']:.0%})", (col_x['Skill']+0.12, i_loc), va='center', ha='center', size=7, color=delta_color)
        ax.add_patch(Rectangle((col_x['Skill']-0.005, i_loc - space/2), 0.075, space,facecolor=cmap(row['C'])))
        
        # Row divider
        ax.axhline(i_loc - space/2, color='black', linewidth=0.3)
        i_loc -= space
    return fig

def create_schedule_results(match_sims,matches,team_ratings,selected_season,selected_team):
    match_sims = match_sims.sort_values('Sim_Date').groupby(['date','Home','Away']).tail(1)
    match_sims = match_sims[['Sim_Date','date','Home','Away','h_exp','a_exp','h_win','d_win','a_win']]
    matches = matches[matches.season == selected_season]
    matches = matches[['date','home_team','away_team','home_score','away_score','home_xg','away_xg','home_p','away_p','home_perf','away_perf','home_xpts','away_xpts']]
    matches = matches.merge(match_sims,left_on=['date','home_team','away_team'],right_on=['date','Home','Away']).drop(columns=['Home','Away'])

    plot_df = matches.merge(
        team_ratings.drop(columns=['Season','A','B']), left_on=['Sim_Date','home_team'], right_on=['Date','Team']).merge(
        team_ratings.drop(columns=['Season','A','B']), left_on=['Sim_Date','away_team'], right_on=['Date','Team'], suffixes=['_H','_A']).drop(
        columns=['Sim_Date','Date_H','Date_A'])
    plot_df['Pre_Pts_H'] = plot_df.h_win * 3 + plot_df.d_win
    plot_df['Pre_Pts_A'] = plot_df.a_win * 3 + plot_df.d_win

    plot_home = plot_df[(plot_df.home_team == selected_team)].reset_index(drop=True)
    plot_home['Loc'] = 'H'
    plot_home = plot_home[['date','Loc','C_H','home_score','away_score','C_A','away_team','h_win','d_win','a_win','Pre_Pts_H','home_xpts','home_xg',
                           'away_xg']].rename(columns={'date':'Date','C_H':'Rtg','C_A':'ORtg','away_team':'Opponent','h_win':'win','d_win':'draw',
                                                       'a_win':'loss','Pre_Pts_H':'Exp','home_xpts':'Perf','home_xg':'xGF','away_xg':'xGA',
                                                       'home_score':'for_score','away_score':'against_score'})
    
    plot_away = plot_df[(plot_df.away_team == selected_team)].reset_index(drop=True)
    plot_away['Loc'] = 'A'
    plot_away = plot_away[['date','Loc','C_A','home_score','away_score','C_H','home_team','a_win','d_win','h_win','Pre_Pts_A','away_xpts','away_xg',
                           'home_xg']].rename(columns={'date':'Date','C_A':'Rtg','C_H':'ORtg','home_team':'Opponent','a_win':'win','d_win':'draw',
                                                       'h_win':'loss','Pre_Pts_A':'Exp','away_xpts':'Perf','away_xg':'xGF','home_xg':'xGA',
                                                       'away_score':'for_score','home_score':'against_score'})
    plot_df = pd.concat((plot_home,plot_away)).sort_values('Date').reset_index(drop=True)
    return plot_df

def create_schedule_results_figure(results,selected_team):
    results.loc[results.for_score.isna(),'score'] = ''
    results.loc[~results.for_score.isna(),'score'] = (results[~results.for_score.isna()].for_score.astype('int').astype('str') + ' - ' + 
                                                      results[~results.against_score.isna()].against_score.astype('int').astype('str'))

    fig_height = max(7, len(results) * 0.2)
    fig, ax = plt.subplots(figsize=(7,fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Column x positions
    col_x = {
        'Date':   0.01,
        'H/A':   0.13,
        'Rtg':    0.16,
        'Score':   0.21,
        'ORtg':0.26,
        'Opponent':0.31,
        '1':  0.56,
        'W':0.59,
        'D':0.64,
        'L':  0.69,
        '2':0.74,
        'Exp':0.77,
        'Per':0.82,
        '3':0.87,
        'xGF':0.90,
        'xGA':0.95}

    # Headers
    header_y = (len(results)+0.5)/(len(results)+1)
    for col, x in col_x.items():
        if (col == 'H/A') | (col == 'Score') | (col == '1') | (col == '2') | (col == '3'):
            pass
        else:
            ha = 'left'
            ax.annotate(col, (x, header_y), va='center', ha=ha, size=7, weight='bold')

    # Row layout
    top = len(results)/(len(results)+1)
    ax.axhline(top, color='black', linewidth=0.8)
    bottom_margin = 1/(len(results)+1)/10
    total_height = top - bottom_margin
    space = total_height / max(4, len(results))
    i_loc = top - space / 2

    # Vertical dividers (between groups)
    for x in col_x.values():
        ax.vlines(x-0.005, bottom_margin, top, color='black', linewidth=0.5)

    for _, row in results.iterrows():
        team_primary = team_colors[selected_team]['home_primary']
        team_text = team_colors[selected_team]['home_secondary']
        opp_primary = team_colors[row['Opponent']]['home_primary']
        opp_text = team_colors[row['Opponent']]['home_secondary']

        if row['Loc'] == 'H':
            loc_background = team_primary
            loc_text = team_text
        else:
            loc_background = opp_primary
            loc_text = opp_text
            
        ax.add_patch(Rectangle((col_x['Date']-0.005,i_loc - space/2),0.12,space,facecolor=team_primary))
        ax.add_patch(Rectangle((col_x['H/A']-0.005,i_loc - space/2),0.03,space,facecolor=loc_background))
        ax.add_patch(Rectangle((col_x['Rtg']-0.005, i_loc - space/2), 0.05, space,facecolor=cmap(row['Rtg']) if pd.notna(row['Rtg']) else 'lightgray'))
        ax.add_patch(Rectangle((col_x['ORtg']-0.005, i_loc - space/2), 0.05, space,facecolor=cmap(row['ORtg']) if pd.notna(row['Rtg']) else 'lightgray'))
        ax.add_patch(Rectangle((col_x['Opponent']-0.005, i_loc - space/2), 0.25, space, facecolor=opp_primary))
        ax.add_patch(Rectangle((col_x['W']-0.005, i_loc - space/2), 0.05, space,facecolor=cmap(norm_w(row['win'])) if pd.notna(row['win']) else 'lightgray'))
        ax.add_patch(Rectangle((col_x['D']-0.005, i_loc - space/2), 0.06, space,facecolor=cmap(norm_w(row['draw'])) if pd.notna(row['draw']) else 'lightgray'))
        ax.add_patch(Rectangle((col_x['L']-0.005, i_loc - space/2), 0.05, space,facecolor=cmap(norm_w(row['loss'])) if pd.notna(row['loss']) else 'lightgray'))
        ax.add_patch(Rectangle((col_x['Exp']-0.005, i_loc - space/2), 0.05, space,facecolor=cmap(norm_p(row['Exp'])) if pd.notna(row['Exp']) else 'lightgray'))
        ax.add_patch(Rectangle((col_x['Per']-0.005, i_loc - space/2), 0.05, space,facecolor=cmap(norm_p(row['Perf'])) if pd.notna(row['Perf']) else 'lightgray'))
        ax.add_patch(Rectangle((col_x['xGF']-0.005, i_loc - space/2), 0.05, space,facecolor=cmap(norm_o(row['xGF'])) if pd.notna(row['xGF']) else 'lightgray'))
        ax.add_patch(Rectangle((col_x['xGA']-0.005, i_loc - space/2), 0.05, space,facecolor=cmap_r(norm_o(row['xGA'])) if pd.notna(row['xGA']) else 'lightgray'))

        
        if row['for_score'] > row['against_score']:
            primary = team_primary
            text = team_text
        elif row['for_score'] < row['against_score']:
            primary = opp_primary
            text = opp_text
        else:
            primary = 'white'
            text = 'black'
        ax.add_patch(Rectangle((col_x['Score']-0.005,i_loc - space/2),0.05,space,facecolor=primary))

        # Text annotations
        ax.annotate(str(row['Date'])[:10], (col_x['Date'], i_loc), va='center', ha='left', size=7,color=team_text,weight='bold')
        ax.annotate(str(row['Loc']), (col_x['H/A']+0.01, i_loc), va='center', ha='center', size=7,color=loc_text,weight='bold')
        ax.annotate(f"{row['Rtg']:.0%}" if pd.notna(row['Rtg']) else '', (col_x['Rtg'], i_loc), va='center', ha='left', size=7)
        ax.annotate(row['score'], (col_x['Score'], i_loc), va='center', ha='left', size=7, fontweight='bold',color=text)
        ax.annotate(f"{row['ORtg']:.0%}" if pd.notna(row['ORtg']) else '', (col_x['ORtg'], i_loc), va='center', ha='left', size=7)
        ax.annotate(row['Opponent'], (col_x['Opponent'], i_loc), va='center', ha='left', size=7,color=opp_text, fontweight='bold')
        ax.annotate(f"{row['win']:.0%}", (col_x['W'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['draw']:.0%}", (col_x['D'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['loss']:.0%}", (col_x['L'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['Exp']:.2f}", (col_x['Exp'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['Perf']:.2f}", (col_x['Per'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['xGF']:.2f}" if pd.notna(row['xGF']) else '',(col_x['xGF'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['xGA']:.2f}" if pd.notna(row['xGA']) else '',(col_x['xGA'], i_loc), va='center', ha='left', size=7)

        # Row divider
        ax.axhline(i_loc - space/2, color='black', linewidth=0.3)
        i_loc -= space
    return fig

def plot_spi_chart(data):
    fig = go.Figure()
    c = np.array(data.C, dtype=float)
    baseline = 0.5
    dates = list(pd.to_datetime(data.Date))
    x_fill = dates + dates[::-1]
    y_green = list(np.where(c >= baseline, c, baseline)) + [baseline] * len(dates)
    fig.add_trace(go.Scatter(
        x=x_fill, y=y_green, fill='toself', mode='none',
        fillcolor='rgba(0,150,0,0.2)', showlegend=False, hoverinfo='skip'
    ))

    y_red = list(np.where(c < baseline, c, baseline)) + [baseline] * len(dates)
    fig.add_trace(go.Scatter(
        x=x_fill, y=y_red, fill='toself', mode='none',
        fillcolor='rgba(200,0,0,0.2)', showlegend=False, hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(x=data.Date, y=np.where(c >= baseline, c, np.nan), mode='lines',
        line=dict(color='darkgreen', width=2.5), showlegend=False, connectgaps=False))
    fig.add_trace(go.Scatter(x=data.Date, y=np.where(c < baseline, c, np.nan), mode='lines',
        line=dict(color='darkred', width=2.5), showlegend=False, connectgaps=False))

    #fig.add_trace(go.Scatter(x=data.Date, y=[baseline] * len(data.Date),mode='none', showlegend=False, hoverinfo='skip'))
    #fig.add_trace(go.Scatter(x=data.Date, y=np.where(c >= baseline, c, baseline),fill='tonexty', mode='none',fillcolor='rgba(0,150,0,0.2)',
    #                         showlegend=False, hoverinfo='skip'))
    #fig.add_trace(go.Scatter(x=data.Date, y=[baseline] * len(data.Date),mode='none', showlegend=False, hoverinfo='skip'))
    #fig.add_trace(go.Scatter(x=data.Date, y=np.where(c < baseline, c, baseline),fill='tonexty', mode='none',fillcolor='rgba(200,0,0,0.2)',
    #    showlegend=False, hoverinfo='skip'))
    
    #fig.add_trace(go.Scatter(x=data.Date, y=np.where(data.C >= 0.5, data.C, np.nan),mode='lines', line=dict(color='darkgreen', width=2.5),showlegend=False, connectgaps=False))
    #fig.add_trace(go.Scatter(x=data.Date, y=np.where(data.C <  0.5, data.C, np.nan),mode='lines', line=dict(color='darkred', width=2.5),showlegend=False, connectgaps=False))
    fig.add_hline(y=0.5, line_dash='dash', line_color='gray', line_width=2)
    for year in pd.to_datetime(data.Date).dt.year.unique():
        fig.add_vline(x=pd.Timestamp(f'{year}-01-01').timestamp()*1000,line_dash='dot', line_color='black', line_width=2)
    fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0),yaxis=dict(range=[0.25, 0.75], tickvals=[i/10 for i in range(3, 8)],tickformat='.0%'),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    season_start = '02-20'
    season_end = '11-09'

    rangebreaks = []
    for year in pd.to_datetime(data.Date).dt.year.unique():
    # Gap before season (Jan 3 → season start)
        rangebreaks.append(dict(bounds=[f'{year}-01-03', f'{year}-{season_start}']))
    # Gap after season (season end → Dec 31)
        rangebreaks.append(dict(bounds=[f'{year}-{season_end}', f'{year}-12-31']))
    fig.update_xaxes(rangebreaks=rangebreaks)
    return fig

def get_crossover_segments(x, a, b):
    segments = []
    diff = a - b
    sign = np.sign(diff)
    
    crossovers = np.where(np.diff(sign) != 0)[0]    
    split_points = [0] + list(crossovers + 1) + [len(x)]
    
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]
        
        seg_x = list(x[start:end])
        seg_a = list(a[start:end])
        seg_b = list(b[start:end])
        
        if i > 0:
            prev = split_points[i] - 1
            curr = split_points[i]
            t = diff[prev] / (diff[prev] - diff[curr])  # linear interpolation
            x_cross = x[prev] + t * (x[curr] - x[prev])
            y_cross = a[prev] + t * (a[curr] - a[prev])
            seg_x = [x_cross] + seg_x
            seg_a = [y_cross] + seg_a
            seg_b = [y_cross] + seg_b
        
        if i < len(split_points) - 2:
            curr = split_points[i + 1] - 1
            nxt = split_points[i + 1]
            t = diff[curr] / (diff[curr] - diff[nxt])
            x_cross = x[curr] + t * (x[nxt] - x[curr])
            y_cross = a[curr] + t * (a[nxt] - a[curr])
            seg_x = seg_x + [x_cross]
            seg_a = seg_a + [y_cross]
            seg_b = seg_b + [y_cross]
        
        a_above = np.mean(seg_a) > np.mean(seg_b)
        segments.append((seg_x, seg_a, seg_b, a_above))
    
    return segments

def plot_offdef_chart(data):
    fig = go.Figure()

    x = np.array(data.Date)
    a = np.array(data.A, dtype=float)
    b = np.array(data.B, dtype=float)

    segments = get_crossover_segments(x, a, b)

    for seg_x, seg_a, seg_b, a_above in segments:
        fill_color = 'rgba(0,180,0,0.4)' if a_above else 'rgba(220,0,0,0.4)'
        # Bottom line first, then top line reversed — creates a closed polygon
        x_fill = seg_x + seg_x[::-1]
        y_fill = seg_b + seg_a[::-1]   # swap if a_above=False? No — always b on bottom, a on top of polygon
        if not a_above:
            y_fill = seg_a + seg_b[::-1]  # a is lower, b is upper
        fig.add_trace(go.Scatter(x=x_fill, y=y_fill,fill='toself', mode='none',fillcolor=fill_color,showlegend=False,hoverinfo='skip'))

    # Draw the two lines on top
    fig.add_hline(y=1.45,line_dash='dash',line_color='grey')
    fig.add_trace(go.Scatter(x=data.Date, y=data.A, mode='lines',line=dict(color='darkgreen', width=2.5), name='Off Rating'))
    fig.add_trace(go.Scatter(x=data.Date, y=data.B, mode='lines',line=dict(color='darkred', width=2.5), name='Def Rating'))

    for year in pd.to_datetime(data.Date).dt.year.unique():
        fig.add_vline(x=pd.Timestamp(f'{year}-01-01').timestamp()*1000,line_dash='dot', line_color='black', line_width=2)
    
    fig.update_layout(height=200,margin=dict(l=0, r=0, t=0, b=0),yaxis=dict(range=[0.45, 2.55], tickvals=[i/10 for i in np.arange(5,26,5)], tickformat='.2f'),
                      plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
    season_start = '02-20'
    season_end = '11-09'

    rangebreaks = []
    for year in pd.to_datetime(data.Date).dt.year.unique():
    # Gap before season (Jan 3 → season start)
        rangebreaks.append(dict(bounds=[f'{year}-01-03', f'{year}-{season_start}']))
    # Gap after season (season end → Dec 31)
        rangebreaks.append(dict(bounds=[f'{year}-{season_end}', f'{year}-12-31']))
    fig.update_xaxes(rangebreaks=rangebreaks)
    return fig

def plot_xg_chart(data):
    fig = go.Figure()
    
    n = 5  # rolling window
    xg = np.array(data.xGF, dtype=float)
    xga = np.array(data.xGA, dtype=float)
    xgd = xg - xga

    # Rolling averages
    def rolling(arr, n):
        result = np.full(len(arr), np.nan)
        for i in range(n - 1, len(arr)):
            result[i] = arr[i - n + 1:i + 1].mean()
        return result

    xg_roll = rolling(xg, n)
    xga_roll = rolling(xga, n)
    xgd_roll = rolling(xgd, n)

    fig.add_trace(go.Bar(x=data.Date, y=xg,marker_color='rgba(0,150,0,0.5)',showlegend=False))
    fig.add_trace(go.Bar(x=data.Date, y=-xga,marker_color='rgba(200,0,0,0.5)',showlegend=False))
    fig.add_trace(go.Scatter(x=data.Date, y=xgd,mode='markers',marker=dict(symbol='diamond', size=6, color='black', opacity=0.6),showlegend=False))
    fig.add_trace(go.Scatter(x=data.Date, y=xg_roll,mode='lines', line=dict(color='darkgreen', width=2),showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=data.Date, y=-xga_roll,mode='lines', line=dict(color='darkred', width=2),showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=data.Date, y=xgd_roll,mode='lines', line=dict(color='black', width=3),showlegend=False, hoverinfo='skip'))

    fig.add_hline(y=0, line_dash='dash',line_color='grey')
    fig.add_hline(y=1.5, line_color='darkgreen', line_dash='dash')
    fig.add_hline(y=-1.5, line_color='darkred', line_dash='dash')

    for year in pd.to_datetime(data.Date).dt.year.unique():
        fig.add_vline(x=pd.Timestamp(f'{year}-01-01').timestamp()*1000,line_dash='dot', line_color='black', line_width=2)
    fig.update_layout(height=200,margin=dict(l=0, r=0, t=0, b=0),yaxis=dict(range=[-4,4], tickvals=list(range(-4, 5)), tickformat='.0f'),
                      plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
    season_start = '02-20'
    season_end = '11-09'

    rangebreaks = []
    for year in pd.to_datetime(data.Date).dt.year.unique():
        rangebreaks.append(dict(bounds=[f'{year}-01-03', f'{year}-{season_start}']))
        rangebreaks.append(dict(bounds=[f'{year}-{season_end}', f'{year}-12-31']))
    fig.update_xaxes(rangebreaks=rangebreaks)
    return fig

standings = pd.read_feather('data/standings.ftr').reset_index()
color_map = pd.DataFrame([['AFC Toronto','#4B0B1A','#FF2929'],
                          ['Calgary Wild FC','#3B1E5E','#C1272D'],
                          ['Halifax Tides FC','#221C35','#00B0B9'],
                          ['Montreal Roses FC','#2D5DA8','#A1283B'],
                          ['Ottawa Rapid FC','#1F5D8C','#4FA7E0'],
                          ['Vancouver Rise FC','#000000','#84AE99']],columns=['team','home_primary','home_secondary']).set_index('team')
team_colors = color_map.to_dict('index')
matches = pd.read_feather('data/matches.ftr')
player_stats = pd.read_feather('data/PlayerStats.ftr')
player_stats['season'] = player_stats.Date.dt.year
team_ratings = pd.read_feather('data/team_ratings.ftr')
team_ratings = team_ratings[['Season','Date']].drop_duplicates().merge(team_ratings[['Season','Team']].drop_duplicates()).merge(
    team_ratings,how='outer').sort_values(['Team','Date'])
team_ratings[['A','B','C']] = team_ratings.groupby(['Season','Team'])[['A','B','C']].ffill()
standings_sims, match_sims = load_standings_sims()

initial_ratings = pd.read_csv('data/Initializations.txt')

# --- MAIN DASHBOARD ---
tab_standings, tab_team = st.tabs(["Standings", "Team Profile"])

with tab_standings:
    col1, col2 = st.columns([2,3])
    # --- COLUMN 1: LEFT ---
    with col1:
        subcol1, subcol2, subcol3 = st.columns([0.5,1,1])
        with subcol1:
            season = sorted(standings_sims['season'].unique(), reverse=True)
            selected_season = st.selectbox("Select Year", options=season, index=0, key="season_picker",label_visibility="collapsed")
        with subcol2:
            dates = sorted(standings_sims[standings_sims['season'] == selected_season]['Sim_Date'].unique(),reverse=True)
            selected_end_date = st.selectbox("Select Date",options=dates,index=0, key="end_date_picker",label_visibility='collapsed')
        with subcol3:
            start_dates = sorted(standings_sims[(standings_sims['season'] == selected_season) & (standings_sims['Sim_Date'] < selected_end_date)]['Sim_Date'].unique(),reverse=True)
            selected_start_date = st.selectbox("Select Relative Date",options=start_dates,index=len(start_dates)-2, key='start_date_picker',label_visibility='collapsed')
        matches_df = create_matches_df(match_sims,matches,team_ratings,selected_season,selected_end_date)
        fig = create_results_figure(matches_df)
        st.markdown("<p style='font-size:14px; font-weight:bold; margin-bottom:2px;'>Results</p>", unsafe_allow_html=True)
        scrollable_plot(fig, height=200)
        fig = create_schedule_figure(matches_df)
        st.markdown("")
        st.markdown("<p style='font-size:14px; font-weight:bold; margin-bottom:2px;'>Schedule</p>", unsafe_allow_html=True)
        scrollable_plot(fig, height=200)

    with col2:
        standings_df = create_standings_file(standings,standings_sims,team_ratings,selected_season,selected_end_date,selected_start_date).sort_values(['P','GD'],ascending=False)
        fig = plot_standings_table(standings_df.drop(columns='season'))
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        st.markdown(f'<img src="data:image/png;base64,{img_base64}" style="width:100%;">', unsafe_allow_html=True)
        plt.close(fig)

        subcol1, subcol2 = st.columns([0.45,0.55])
        with subcol1:
            fig = plot_ratings_scatter(standings_df.drop(columns='season'), team_colors)
            st.plotly_chart(fig, use_container_width=False)
        with subcol2:
            fig_heatmap = plot_position_heatmap(standings_sims, standings_df, selected_end_date, team_colors)
            st.plotly_chart(fig_heatmap)
            st.markdown("")
            mvp_df = create_player_mvps(player_stats,matches_df,int(selected_season),selected_end_date)
            st.markdown("<p style='font-size:14px; font-weight:bold; margin-bottom:2px;'>Best Players</p>", unsafe_allow_html=True)
            fig = create_mvp_figure(mvp_df)
            scrollable_plot(fig, height=200)

with tab_team:
    col1, col2 = st.columns([2,3])
    with col1:
        subcol1, subcol2, subcol3 = st.columns([1.5,1.5,1.5])
        with subcol1:
            teams = sorted(standings.F_team.unique())
            selected_team = st.selectbox('Select Team',options=teams,key='team_picker',label_visibility='collapsed')
        with subcol2:
            season = sorted(standings['season'].unique(), reverse=True)
            selected_season = st.selectbox("Select Year", options=season, index=0, key="season_picker2",label_visibility="collapsed")
            schedule_results = create_schedule_results(match_sims,matches,team_ratings,selected_season,selected_team)
        with subcol3:
            selected_visual_type = st.selectbox('Select Type',options=['Net Rtg','Off/Def','xG/xGA'],label_visibility='collapsed')
            multi_standings = create_multi_year_standings(team_ratings,standings)
            multi_standings = multi_standings[multi_standings.Team == selected_team]
        fig = plot_history_table(multi_standings)
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        st.markdown(f'<img src="data:image/png;base64,{img_base64}" style="width:100%;">', unsafe_allow_html=True)
        plt.close(fig)
        fig = create_schedule_results_figure(schedule_results,selected_team)
        scrollable_plot(fig, height=390)
    with col2:
        if selected_visual_type == 'Net Rtg':
            fig = plot_spi_chart(team_ratings[team_ratings.Team == selected_team])
            st.plotly_chart(fig, use_container_width=True)
        elif selected_visual_type == 'Off/Def':
            fig = plot_offdef_chart(team_ratings[team_ratings.Team == selected_team])
            st.plotly_chart(fig, use_container_width=True)
        else:
            schedule_results = []
            for i in season:
                schedule_results.append(create_schedule_results(match_sims,matches,team_ratings,i,selected_team))
            schedule_results = pd.concat(schedule_results)
            fig = plot_xg_chart(schedule_results.sort_values('Date').reset_index(drop=True))
            st.plotly_chart(fig, use_container_width=True)