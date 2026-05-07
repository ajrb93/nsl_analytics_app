import pandas as pd
import numpy as np
import os
import pickle

def define_dates_sims(results):
    results = results[~results.home_score.isna()]
    past_dates = pd.to_datetime(results.date).drop_duplicates().sort_values().dt.date.values
    years = pd.to_datetime(results.date).dt.year.drop_duplicates().sort_values()
    for year in years:
        past_dates = np.append(past_dates,pd.to_datetime(str(year)+'-01-01').date())
        past_dates = np.append(past_dates,pd.to_datetime(str(year)+'-01-02').date())
    past_dates = np.sort(past_dates)

    if past_dates[-2] == pd.to_datetime(str(years.iloc[-1])+'-01-01').date():
        past_dates = past_dates[:-2]
    return past_dates

def find_matches(date,season,temp_matches):
    temp_matches = temp_matches.copy()
    temp_matches['GP'] = 1
    temp_matches = temp_matches[temp_matches.season == season]
    temp_results = temp_matches[pd.to_datetime(temp_matches.date).dt.date <= date][['date','home_team','away_team','home_score','away_score','GP']]
    temp_schedule = temp_matches[pd.to_datetime(temp_matches.date).dt.date > date][['date','home_team','away_team','GP']]
    return temp_schedule, temp_results

def find_ratings_all(ratings, season, date):
    result = {}
    for team, team_data in ratings.items():
        if season not in team_data:
            continue
        season_data = team_data[season]
        valid_dates = [d for d in season_data if d <= date]
        if not valid_dates:
            continue
        latest_date = max(valid_dates)
        result.setdefault(team, {})[season] = {latest_date: season_data[latest_date]}
    return result

def find_ratings(ratings,team,season,date):
    season_data = ratings[team][season]
    valid_dates = [d for d in season_data.keys() if d <= date]
    latest_date = max(valid_dates)
    result = season_data[latest_date]
    return result

def prepare_rating_arrays(temp_ratings, temp_schedule, temp_results, season, date):
    teams = pd.concat([temp_schedule['home_team'],temp_schedule['away_team'],temp_results['home_team'],temp_results['away_team']]).unique()
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    n_teams = len(teams)
    ratings_array = np.zeros((n_teams, 2))
    for team in teams:
        rating = find_ratings(temp_ratings, team, season, date)
        ratings_array[team_to_idx[team]] = rating[:2]
    return ratings_array, team_to_idx, teams

def simulate_season_vectorized(schedule_home_idx, schedule_away_idx, initial_ratings,temp_goals, temp_home_field, update_rate):
    n_matches = len(schedule_home_idx)
    ratings = initial_ratings.copy()
    
    home_goals = np.zeros(n_matches, dtype=int)
    away_goals = np.zeros(n_matches, dtype=int)
    
    for i in range(n_matches):
        home_idx = schedule_home_idx[i]
        away_idx = schedule_away_idx[i]
        home_off, home_def = ratings[home_idx, 0], ratings[home_idx, 1]
        away_off, away_def = ratings[away_idx, 0], ratings[away_idx, 1]
        
        temp_home_xg = temp_goals / 2 * np.exp(np.log(home_off) + np.log(away_def) + temp_home_field / 2)
        temp_away_xg = temp_goals / 2 * np.exp(np.log(home_def) + np.log(away_off) - temp_home_field / 2)
        h_g = np.random.poisson(temp_home_xg)
        a_g = np.random.poisson(temp_away_xg)
        home_goals[i] = h_g
        away_goals[i] = a_g
        home_perf = temp_home_xg * 0.7 + h_g * 0.3
        away_perf = temp_away_xg * 0.7 + a_g * 0.3
        
        home_off_perf = np.log(home_perf / temp_home_xg)
        away_off_perf = np.log(away_perf / temp_away_xg)
        home_def_perf = away_off_perf * -1 
        away_def_perf = home_off_perf * -1
        ratings[home_idx, 0] = home_off + update_rate * home_off_perf
        ratings[home_idx, 1] = home_def - update_rate * home_def_perf
        ratings[away_idx, 0] = away_off + update_rate * away_off_perf
        ratings[away_idx, 1] = away_def - update_rate * away_def_perf
    
    return home_goals, away_goals

def fast_table_from_goals(home_goals, away_goals, home_teams, away_teams,results_summary, team_to_idx):
    n_teams = len(team_to_idx)
    points = np.zeros(n_teams)
    goals_for = np.zeros(n_teams)
    goals_against = np.zeros(n_teams)
    
    home_wins = (home_goals > away_goals).astype(int)
    away_wins = (away_goals > home_goals).astype(int)
    draws = (home_goals == away_goals).astype(int)
    
    for i in range(len(home_goals)):
        h_idx = home_teams[i]
        a_idx = away_teams[i]
        
        points[h_idx] += home_wins[i] * 3 + draws[i]
        points[a_idx] += away_wins[i] * 3 + draws[i]
        
        goals_for[h_idx] += home_goals[i]
        goals_for[a_idx] += away_goals[i]
        goals_against[h_idx] += away_goals[i]
        goals_against[a_idx] += home_goals[i]
    
    if results_summary is not None:
        for team, idx in team_to_idx.items():
            if team in results_summary.index:
                points[idx] += results_summary.loc[team, 'Points']
                goals_for[idx] += results_summary.loc[team, 'F']
                goals_against[idx] += results_summary.loc[team, 'A']
    
    goal_diff = goals_for - goals_against
    
    ranks = np.zeros(n_teams, dtype=int)
    for rank, idx in enumerate(np.lexsort((-goals_for, -goal_diff, -points)), 1):
        ranks[idx] = rank
    
    return ranks, points, goals_for, goals_against, goal_diff

def summarize_matches(results):
    summary = pd.concat((
        results[['home_team','home_score','away_score']].rename(columns={'home_team':'Team','home_score':'F','away_score':'A'}),
        results[['away_team','away_score','home_score']].rename(columns={'away_team':'Team','away_score':'F','home_score':'A'})))

    summary['Points'] = ((summary.F > summary.A).astype('int') * 3 + (summary.F == summary.A).astype('int'))
    summary['Goal_D'] = summary.F - summary.A
    summary = summary.groupby('Team').sum().sort_values(['Points','Goal_D','F'],ascending=[False,False,False])
    summary['rank'] = summary[['Points','Goal_D','F']].apply(tuple,axis=1).rank(method='dense',ascending=False).astype('int')
    return summary

def simulate_individual_matches(temp_schedule, ratings_array, team_to_idx,temp_goals, temp_home_field, n_sims):
    n_matches = len(temp_schedule)
    
    home_exp = np.zeros(n_matches)
    away_exp = np.zeros(n_matches)
    home_wins = np.zeros(n_matches)
    away_wins = np.zeros(n_matches)
    draws = np.zeros(n_matches)

    all_match_stats = []
    for i, (idx, row) in enumerate(temp_schedule.iterrows()):
        temp_date = row['date']
        temp_home = row['home_team']
        temp_away = row['away_team']
        
        home_idx = team_to_idx[temp_home]
        away_idx = team_to_idx[temp_away]
        home_off, home_def = ratings_array[home_idx, 0], ratings_array[home_idx, 1]
        away_off, away_def = ratings_array[away_idx, 0], ratings_array[away_idx, 1]   
        temp_home_exp = temp_goals / 2 * np.exp(np.log(home_off) + np.log(away_def) + temp_home_field / 2)
        temp_away_exp = temp_goals / 2 * np.exp(np.log(home_def) + np.log(away_off) - temp_home_field / 2)
        temp_home_arr = np.random.poisson(temp_home_exp, n_sims)
        temp_away_arr = np.random.poisson(temp_away_exp, n_sims)
        temp_home_win = np.sum(temp_home_arr > temp_away_arr) / n_sims
        temp_away_win = np.sum(temp_home_arr < temp_away_arr) / n_sims
        temp_draw = np.sum(temp_home_arr == temp_away_arr) / n_sims
        
        home_goals_unique, home_goals_counts = np.unique(temp_home_arr, return_counts=True)
        away_goals_unique, away_goals_counts = np.unique(temp_away_arr, return_counts=True)
        temp_home_dist = dict(zip(home_goals_unique, home_goals_counts / n_sims))
        temp_away_dist = dict(zip(away_goals_unique, away_goals_counts / n_sims))
        
        temp_stats = pd.DataFrame({'date': [temp_date],'Home': [temp_home],'Away': [temp_away],'h_exp': [temp_home_exp],'a_exp': [temp_away_exp],
                                   'h_win': [temp_home_win],'d_win': [temp_draw],'a_win': [temp_away_win]})
        
        for goal, prob in temp_home_dist.items():
            temp_stats[f'h_{goal}'] = prob
        
        for goal, prob in temp_away_dist.items():
            temp_stats[f'a_{goal}'] = prob
        all_match_stats.append(temp_stats)
        
    return pd.concat(all_match_stats, ignore_index=True)

def simulate_matchups(sim_dates,matches,team_ratings,total_goals,home_field,n_sims):
    for date in sim_dates:
        season = date.year
        
        temp_schedule, temp_results = find_matches(date, season, matches)
        temp_ratings = find_ratings_all(team_ratings, season, date)
        temp_goals = total_goals[season]
        temp_home_field = home_field[season]
        
        if len(temp_schedule) != 0:
            ratings_array, team_to_idx, teams = prepare_rating_arrays(temp_ratings, temp_schedule, temp_results, season, date)
            match_sims = simulate_individual_matches(temp_schedule, ratings_array, team_to_idx,temp_goals, temp_home_field, n_sims)
            match_sims.insert(0, 'Sim_Date', date)
            
            match_sims.to_feather(f'data/Sim_States/{date}_matches.ftr')

def simulate_season(sim_dates,matches,total_goals,home_field,n_sims,update_rate,team_ratings):
    for date in sim_dates:
        print(date)
        season = date.year

        temp_schedule, temp_results = find_matches(date, season, matches)
        temp_ratings = find_ratings_all(team_ratings, season, date)
        temp_goals = total_goals[season]
        temp_home_field = home_field[season]
        
        if len(temp_schedule) != 0:
            ratings_array, team_to_idx, teams = prepare_rating_arrays(temp_ratings, temp_schedule, temp_results, season, date)
            schedule_home_idx = temp_schedule['home_team'].map(team_to_idx).values
            schedule_away_idx = temp_schedule['away_team'].map(team_to_idx).values
            
            results_summary = summarize_matches(temp_results) if len(temp_results) > 0 else None
            all_ranks = np.zeros((n_sims, len(teams)), dtype=int)
            all_points = np.zeros((n_sims, len(teams)))
            all_gf = np.zeros((n_sims, len(teams)))
            all_ga = np.zeros((n_sims, len(teams)))
            all_gd = np.zeros((n_sims, len(teams)))
            
            for sim in range(n_sims):
                home_goals, away_goals = simulate_season_vectorized(schedule_home_idx, schedule_away_idx, ratings_array,temp_goals,temp_home_field, update_rate)
                ranks, points, gf, ga, gd = fast_table_from_goals(home_goals,away_goals,schedule_home_idx, schedule_away_idx,results_summary,team_to_idx)
                all_ranks[sim] = ranks
                all_points[sim] = points
                all_gf[sim] = gf
                all_ga[sim] = ga
                all_gd[sim] = gd
            
            sim_results = pd.DataFrame({'Points': all_points.mean(axis=0),'F': all_gf.mean(axis=0),'A': all_ga.mean(axis=0),'Goal_D': all_gd.mean(axis=0),
                                        'rank': all_ranks.mean(axis=0)}, index=teams)
            for rank in range(1, len(teams) + 1):
                sim_results[str(rank)] = (all_ranks == rank).mean(axis=0)

        else:
            results_summary = summarize_matches(temp_results)
            sim_results = results_summary.copy()
            n_teams = len(results_summary)
            for rank in range(1, n_teams + 1):
                sim_results[str(rank)] = 0.0
            for team in results_summary.index:
                actual_rank = results_summary.loc[team, 'rank']
                sim_results.loc[team, str(actual_rank)] = 1.0
                
        sim_results.to_feather(f'data/Sim_States/{date}.ftr')

matches = pd.read_feather('data/matches.ftr')
matches.match_id = matches.match_id.astype('str')
matches.date = pd.to_datetime(matches.date).dt.date
matches = matches[matches['round'] == 'Regular Season']

season_mapping = matches[['season','date']].drop_duplicates()
season_mapping = season_mapping.set_index('date').to_dict()['season']

with open('data/total_goals.pkl', 'rb') as f:
    tg = pickle.load(f)

with open('data/home_field.pkl', 'rb') as f:
    hf = pickle.load(f)

with open('data/team_ratings.pkl', 'rb') as f:
    team_ratings = pickle.load(f)

past_dates = define_dates_sims(matches)
prev_sims = os.listdir('data/Sim_States')
prev_sims = list(filter(lambda k: '_matches' not in k, prev_sims))
prev_sims = list(filter(lambda k: '.ftr' in k, prev_sims))
prev_sims = [s.replace('.ftr', '') for s in prev_sims]
prev_sims = pd.to_datetime(prev_sims).date
sim_dates = sorted(list(set(past_dates) - set(prev_sims)))
print(len(past_dates),len(prev_sims),len(sim_dates))

n_sims = 10000
update_rate = 2/25

simulate_season(sim_dates,matches,tg,hf,n_sims,update_rate,team_ratings)
simulate_matchups(sim_dates,matches,team_ratings,tg,hf,n_sims)