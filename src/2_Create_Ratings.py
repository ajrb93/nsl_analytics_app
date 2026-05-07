import pandas as pd
import numpy as np
from scipy.stats import skellam
from scipy.optimize import minimize_scalar
import pickle

def calculate_expected_points(row,n_sims=10000):
    home_probs = np.tile(row.home_xg_l, n_sims)
    away_probs = np.tile(row.away_xg_l, n_sims)
    home_random = np.random.rand(len(home_probs))
    away_random = np.random.rand(len(away_probs))
    home_goals = (home_random < home_probs).reshape(n_sims, -1).sum(axis=1)
    away_goals = (away_random < away_probs).reshape(n_sims, -1).sum(axis=1)
    home_points = np.where(home_goals > away_goals, 3,np.where(home_goals == away_goals, 1, 0))
    away_points = np.where(away_goals > home_goals, 3,np.where(home_goals == away_goals, 1, 0))

    return home_points.mean() if row.finished == True else np.nan, away_points.mean() if row.finished == True else np.nan

def create_xg(df):
    total_goals = df.home_score.sum() + df.away_score.sum()
    total_shots = df.home_ib.sum() + df.home_ob.sum() + df.away_ib.sum() + df.away_ob.sum()
    total_ob = df.home_ob.sum() + df.away_ob.sum()
    total_ib = df.home_ib.sum() + df.away_ib.sum() - df.home_pen.sum() - df.away_pen.sum()
    total_pen = df.home_pen.sum() + df.away_pen.sum()
    pen = 0.72
    xg_shot = total_goals / total_shots
    ob_xg_shot = 0.035
    ib_xg_shot = np.round((total_goals - total_ob * ob_xg_shot + total_pen * pen) / total_ib,3)
    xg_parameters = [xg_shot,pen,ob_xg_shot,ib_xg_shot]
    df['home_xg'] = df.home_pen * xg_parameters[1] + (df.home_ib-df.home_pen) * xg_parameters[3] + df.home_ob * xg_parameters[2]
    df['away_xg'] = df.away_pen * xg_parameters[1] + (df.away_ib-df.away_pen) * xg_parameters[3] + df.away_ob * xg_parameters[2]
    df['home_xg_l'] = [np.repeat([xg_parameters[1],xg_parameters[2],xg_parameters[3]],[a,b,c-a]) if pd.notna(a) and pd.notna(b) and pd.notna(c) else 
                       np.nan for a,b,c in zip(df.home_pen,df.home_ob,df.home_ib)]
    df['away_xg_l'] = [np.repeat([xg_parameters[1],xg_parameters[2],xg_parameters[3]],[a,b,c-a]) if pd.notna(a) and pd.notna(b) and pd.notna(c) else 
                       np.nan for a,b,c in zip(df.away_pen,df.away_ob,df.away_ib)]
    df['home_p'] = (df.home_score > df.away_score).astype('int') * 3 + (df.home_score == df.away_score).astype('int') 
    df['away_p'] = (df.home_score < df.away_score).astype('int') * 3 + (df.home_score == df.away_score).astype('int') 
    df[['home_xpts', 'away_xpts']] = df.apply(lambda row: calculate_expected_points(row, n_sims=10000),axis=1, result_type="expand")
    df['home_perf'] = df.home_xg * 0.7 + df.home_score * 0.3
    df['away_perf'] = df.away_xg * 0.7 + df.away_score * 0.3
    df['total_goals'] = df.home_score + df.away_score
    df['home_field'] = df.home_score - df.away_score
    return df

def calculate_standings(fixtures):
    fixtures['GP'] = (fixtures.finished == True).astype('int')
    temp_home = fixtures[fixtures['round'] == 'Regular Season'][['season','home_team','away_team','GP','home_score','away_score','home_p',
                                                                 'home_xg','away_xg','home_xpts']].reset_index(drop=True)
    temp_home.columns = temp_home.columns.str.replace('home','F').str.replace('away','A')
    temp_home['HR'] = (temp_home.GP == 1).astype('int') * 1
    temp_away = fixtures[fixtures['round'] == 'Regular Season'][['season','home_team','away_team','GP','home_score','away_score','away_p',
                                                                 'home_xg','away_xg','away_xpts']].reset_index(drop=True)
    temp_away.columns = temp_away.columns.str.replace('away','F').str.replace('home','A')
    temp_away['HR'] = (temp_away.GP == 1).astype('int') * -1
    results_t = pd.concat((temp_home,temp_away))
    standings = results_t.groupby(['season','F_team']).sum(numeric_only=True)
    standings['PPG'] = np.round(standings.F_p / standings.GP,2)
    standings['oRTG'] = np.round(standings.F_score / standings.GP * 0.3 + standings.F_xg / standings.GP * 0.7,2)
    standings['dRTG'] = np.round(standings.A_score / standings.GP * 0.3 + standings.A_xg / standings.GP * 0.7,2)
    standings['hRTG'] = np.round(standings.HR / standings.GP * (temp_home.F_xg.mean() - temp_away.F_xg.mean()) * -1,2)
    standings['nRTG'] = standings.oRTG - standings.dRTG + standings.hRTG
    standings = standings.sort_values('nRTG',ascending=False)
    return standings

def calculate_parameters(fixtures):
    hf = {}
    tg = {}
    for date in matches.dropna(subset='home_score').groupby('season').date.last().values:
        temp = matches[matches.date <= date].tail(70)
        season = temp.season.tail(1).values[0]
        tg[season] = temp.total_goals.mean()
        hf[season] = temp.home_field.mean()
    return hf, tg

def team_rating(xg,xga):
    win_rate = (1 - skellam.cdf(0,xg,xga)) + skellam.pmf(0,xg,xga) / 2
    return np.round(win_rate,6)

def define_dates_ratings(results):
    results = results[~results.home_score.isna()]
    past_dates = pd.to_datetime(results.date).drop_duplicates().sort_values().dt.date.values
    years = pd.to_datetime(results.date).dt.year.drop_duplicates().sort_values()
    for year in years:
        past_dates = np.append(past_dates,pd.to_datetime(str(year)+'-01-01').date())
    past_dates = np.sort(past_dates)

    if past_dates[-1] == pd.to_datetime(str(years.iloc[-1])+'-01-01').date():
        past_dates = past_dates[:-1]
    
    return past_dates

def adjust_xg_xga(xg, xga, target_rating):
    delta_min = -xg  
    delta_max = xga     
    def loss(delta):
        return (team_rating(xg + delta, xga - delta) - target_rating)**2
    res = minimize_scalar(loss, bounds=(delta_min, delta_max), method='bounded')
    delta = res.x
    return (xg + delta)[0], (xga - delta)[0], target_rating[0]

def add_initial_season_ratings(team,team_ratings,season,team_initializations,transfer_vals):
    temp_season = season
    season_start_date = pd.to_datetime(str(temp_season) + '-01-01').date()
    season_transfer_date = pd.to_datetime(str(temp_season) + '-01-02').date()
    previous_season = int(season)-1

    if team in team_ratings and previous_season in team_ratings[team]:
        season_data = team_ratings[team][previous_season]
        latest_date = sorted(season_data.keys())[-1]
        starting_rankings = season_data[latest_date]
    else:
        filtered = team_initializations[(team_initializations.team == team) &(team_initializations.season == season)]
                                        
        if filtered.empty:
            raise ValueError(f"No initialization found for {team} in {season}")
        else:
            starting_rankings = team_initializations[((team_initializations.team == team) & 
                                                    (team_initializations.season == season))][['ORtg','DRtg','WinRate']].iloc[0].values
    
    rating_adjustment = transfer_vals[(transfer_vals.team == team) & (transfer_vals.season == season)].Value.values[0]
    adjusted_rankings = np.round([starting_rankings[2] * 0.67 + rating_adjustment * 0.33],6)
    adjusted_rankings = list(adjust_xg_xga(starting_rankings[0],starting_rankings[1],adjusted_rankings))
    
    if team not in team_ratings:
        team_ratings[team] = {}
    if season not in team_ratings[team]:
        team_ratings[team][season] = {}

    team_ratings[team][season][season_start_date] = list(starting_rankings)
    team_ratings[team][season][season_transfer_date] = adjusted_rankings

def update_ratings(row,team_ratings,season,total_goals,home_field,update_rate):
    temp_date = row['date']
    temp_home = row['home_team']
    temp_away = row['away_team']
    temp_home_perf = row['home_perf']
    temp_away_perf = row['away_perf']

    temp_home_rating = team_ratings[temp_home][season][sorted(team_ratings[temp_home][season].keys())[-1]]
    temp_away_rating = team_ratings[temp_away][season][sorted(team_ratings[temp_away][season].keys())[-1]]

    temp_home_exp = total_goals / 2 * np.exp(np.log(temp_home_rating[0]) + np.log(temp_away_rating[1]) + home_field / 2)
    temp_home_off_perf = np.log(temp_home_perf / temp_home_exp)
    temp_away_def_perf = temp_home_off_perf * -1
    temp_away_exp = total_goals / 2 * np.exp(np.log(temp_away_rating[0]) + np.log(temp_home_rating[1]) - home_field / 2)
    temp_away_off_perf = np.log(temp_away_perf / temp_away_exp)
    temp_home_def_perf = temp_away_off_perf * -1    

    temp_home_rating_adj = [temp_home_rating[0] + update_rate * temp_home_off_perf,
                            temp_home_rating[1] - update_rate * temp_home_def_perf]
    temp_home_rating_adj.append(team_rating(temp_home_rating_adj[0]*total_goals/2,temp_home_rating_adj[1]*total_goals/2))
    temp_away_rating_adj = [temp_away_rating[0] + update_rate * temp_away_off_perf,
                            temp_away_rating[1] - update_rate * temp_away_def_perf]
    temp_away_rating_adj.append(team_rating(temp_away_rating_adj[0]*total_goals/2,temp_away_rating_adj[1]*total_goals/2))

    team_ratings[temp_home][season][temp_date] = temp_home_rating_adj
    team_ratings[temp_away][season][temp_date] = temp_away_rating_adj    

def normalize_ratings(team_ratings, target_date):
    rows = []
    for team, seasons in team_ratings.items():
        for season, dates in seasons.items():
            if target_date in dates:
                off, deff, rtg = dates[target_date]
                rows.append({'Team': team,'Season': season,'Date': target_date,'Off': off,'Def': deff,'Rtg': rtg})
    if not rows:
        print(f"No ratings found for {target_date}")
        return team_ratings
    df = pd.DataFrame(rows)

    mean_off = df['Off'].mean()
    mean_def = df['Def'].mean()
    target_mean = (mean_off + mean_def) / 2
    off_adjustment = mean_off - target_mean
    def_adjustment = mean_def - target_mean
    df['Off'] -= off_adjustment
    df['Def'] -= def_adjustment
    df['Rtg'] = team_rating(df.Off,df.Def)
    
    for _, row in df.iterrows():
        team = row['Team']
        season = row['Season']
        team_ratings[team][season][target_date] = [row['Off'], row['Def'], row['Rtg']]
    return team_ratings

def calculate_ratings(past_dates,transfer_vals,team_initializations,season_mapping,total_goals,home_field,results,update_rate):
    team_ratings = {}
    for date in past_dates:
        if date.month == 1 and date.day == 1:
            season = date.year
            print(season)
            for team in transfer_vals[transfer_vals.season == season].team.values:
                add_initial_season_ratings(team,team_ratings,season,team_initializations,transfer_vals)    
            normalize_ratings(team_ratings, pd.to_datetime(f"{date.year}-01-01").date())
            normalize_ratings(team_ratings, pd.to_datetime(f"{date.year}-01-02").date())
        else:
            season = season_mapping[pd.to_datetime(date).date()]
            total_goals_season = total_goals[season]
            home_field_season = home_field[season]
            matches_temp = results[results.date == date]

            for idx, row in matches_temp.iterrows():
                update_ratings(row,team_ratings,season,total_goals_season,home_field_season,update_rate)
    return team_ratings

def clean_team_ratings(team_ratings):
    rows = []
    for team, seasons in team_ratings.items():
        for season, dates in seasons.items():
            for date, values in dates.items():
                rows.append({
                    'Team': team,
                    'Season': season,
                    'Date': date,
                    'A': values[0],
                    'B': values[1],
                    'C': values[2]
                })
    clean_team_ratings = pd.DataFrame(rows)
    return clean_team_ratings

matches = pd.read_csv('data/Matches.csv').drop(columns='Unnamed: 0')
matches.match_id = matches.match_id.astype('str')
matches.date = pd.to_datetime(matches.date).dt.date

matches = create_xg(matches)
standings = calculate_standings(matches)
hf, tg = calculate_parameters(matches)

initializations = pd.read_csv('data/Initializations.txt')
initializations['WinRate'] = initializations.apply(lambda row: team_rating(row['ORtg'], row['DRtg']), axis=1)
transfer_vals = pd.read_csv('data/TransferMarkt.txt')
transfer_vals['mean'] = transfer_vals.groupby('season').Value.transform('mean')
transfer_vals['std'] = transfer_vals.groupby('season').Value.transform('std')
transfer_vals.Value = (transfer_vals.Value - transfer_vals['mean'])/transfer_vals['std']
transfer_vals.Value = (transfer_vals.Value * 0.3 + 1.5)/3

season_mapping = matches[['season','date']].drop_duplicates()
season_mapping = season_mapping.set_index('date').to_dict()['season']

matches.to_feather('data/matches.ftr')
standings.to_feather('data/standings.ftr')
with open("data/home_field.pkl", "wb") as f:
    pickle.dump(hf, f)
with open("data/total_goals.pkl", "wb") as f:
    pickle.dump(tg, f)

team_ratings = calculate_ratings(define_dates_ratings(matches),transfer_vals,initializations,season_mapping,tg,hf,matches,2/25)
with open("data/team_ratings.pkl", "wb") as f:
    pickle.dump(team_ratings, f)
clean_team_ratings(team_ratings).to_feather('data/team_ratings.ftr')