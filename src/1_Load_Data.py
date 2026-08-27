import pandas as pd
import numpy as np
import requests
import unicodedata

def refresh_matches(year):
    original_df = pd.read_csv('data/Matches.csv').drop(columns=['Unnamed: 0','status'])
    original_df = original_df[original_df.finished == True].drop(columns='finished')
    original_df.match_id = original_df.match_id.astype('str')
    original_df.date = pd.to_datetime(original_df.date).dt.date

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    url = 'https://www.fotmob.com/api/data/leagues?id=10872&season='+str(year)
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        fixtures = response.json()['fixtures']['allMatches']
    else:
        print('data refresh failed')

    fixtures_list = []
    for match in fixtures:
        fixture = {
            'match_id': match.get('id'),
            'date': match.get('status', {}).get('utcTime'),
            'round': match.get('round'),
            'home_team': match.get('home', {}).get('name'),
            'away_team': match.get('away', {}).get('name'),
            'status': match.get('status', {}).get('started'),
            'finished': match.get('status', {}).get('finished')}
        fixtures_list.append(fixture)
    
    df = pd.DataFrame(fixtures_list)
    df.date = pd.to_datetime(df.date,format='mixed').dt.tz_convert('EST').dt.date
    df = original_df.merge(df,how='outer',on=['match_id','date','home_team','away_team','round']).sort_values(['date','match_id'])
    df[['status','finished']] = df[['status','finished']].fillna(True)
    df['season'] = pd.to_datetime(df.date).dt.year
    df["home_team"] = df["home_team"].apply(lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii"))
    df["away_team"] = df["away_team"].apply(lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii"))
    return df

fixtures = refresh_matches(2026)
fixtures.to_csv('data/Matches.csv')

player_stats = pd.read_csv('data/PlayerStats.txt')
player_stats.Date = pd.to_datetime(player_stats.Date, unit='D', origin='1899-12-30')
player_stats.P = player_stats.P.fillna('')
test1 = player_stats.groupby(['Date','Team'])[['MIN','RC']].sum()
print(test1[test1.MIN != 900].sort_values('MIN'))
rc_rate = player_stats[player_stats.RC == 1]
rc_rate = (rc_rate.GF.sum() - rc_rate.GA.sum()) / rc_rate.MIN.sum() * 90
player_stats.loc[player_stats.RC == 1,('MIN','GF','GA','In','Out')] = 0
player_stats = player_stats.groupby(['Date','Type','Team','Player','P']).sum(numeric_only=True).reset_index()

goalie_stats = pd.read_csv('data/GoalieStats.txt')
goalie_stats.Date = pd.to_datetime(goalie_stats.Date, unit='D', origin='1899-12-30')
test2 = goalie_stats.groupby(['Date','Team']).MIN.sum().reset_index()
print(test2[test2.MIN != 90].sort_values('MIN'))

goalie_rate = goalie_stats.GA.sum()/(goalie_stats.SV.sum() + goalie_stats.GA.sum() + goalie_stats.PCH.sum() + goalie_stats.DS.sum() / 2)
goalie_stats['SVAA'] = (goalie_stats.GA + goalie_stats.SV + goalie_stats.PCH + goalie_stats.DS / 2) * goalie_rate - goalie_stats.GA - goalie_stats.OG
goalie_stats['P'] = 'G'
goalie_stats = goalie_stats[['Date','Type','Team','Goalkeeper','MIN','SVAA','P']].rename(columns={'Goalkeeper':'Name','SVAA':'Rtg'})

g_val = 1
a_val = player_stats.A.sum() / player_stats.G.sum()
osa_val = player_stats.G.sum() / player_stats.OSA.sum()
tsa_val = player_stats.OSA.sum() / player_stats.TSA.sum()* osa_val
tp_val = player_stats.G.sum() / player_stats.TP.sum()
tcr_val = player_stats.OSA.sum() / player_stats.TCR.sum() * osa_val
tch_val = player_stats.G.sum() / player_stats.TCH.sum()
tt_val = player_stats.G.sum() / player_stats.TT.sum()

vals = [g_val,a_val,osa_val,tsa_val,tp_val,tch_val,tcr_val]
vals /= np.sum(vals)

player_stats['xGF'] = (player_stats.G * vals[0] + player_stats.A * vals[1] + player_stats.TSA * vals[3] + player_stats.OSA * vals[2] + 
                       player_stats.TP * vals[4] + player_stats.TCH * vals[5] + player_stats.TCR * vals[6])
player_stats['xGF'] = player_stats.xGF - player_stats.xGF.sum() / player_stats.MIN.sum() * player_stats.MIN
player_stats.xGF += player_stats.RC * rc_rate * (90- player_stats.Out) / 90 / 2

f_adj = (player_stats.groupby('P').sum(numeric_only=True).loc['F'].xGF / player_stats.groupby('P').sum(numeric_only=True).loc['F'].MIN)
m_adj = (player_stats.groupby('P').sum(numeric_only=True).loc['M'].xGF / player_stats.groupby('P').sum(numeric_only=True).loc['M'].MIN)
d_adj = (player_stats.groupby('P').sum(numeric_only=True).loc['D'].xGF / player_stats.groupby('P').sum(numeric_only=True).loc['D'].MIN)
player_stats.loc[player_stats.P == 'F','xGF'] -= f_adj * player_stats.MIN
player_stats.loc[player_stats.P == 'M','xGF'] -= m_adj * player_stats.MIN
player_stats.loc[player_stats.P == 'D','xGF'] -= d_adj * player_stats.MIN
player_stats['xGF'] = (player_stats.GF - player_stats.groupby(['Date','Team']).GF.max().mean() / 90 * player_stats.MIN) * 0.15 + player_stats.xGF * 0.85

player_stats['xGA'] = (player_stats.groupby(['Date','Team']).GF.max().mean() / 90 * player_stats.MIN - player_stats.TT * tt_val - 
                       player_stats.TCH * tch_val + player_stats.F * tsa_val + player_stats.YC * rc_rate / 20)
player_stats['xGA'] = (player_stats.xGA - player_stats.xGA.sum() / player_stats.MIN.sum() * player_stats.MIN)
player_stats.xGA -= player_stats.RC * rc_rate * (90- player_stats.Out) / 90 / 2

f_adj = (player_stats.groupby('P').sum(numeric_only=True).loc['F'].xGA / player_stats.groupby('P').sum(numeric_only=True).loc['F'].MIN)
m_adj = (player_stats.groupby('P').sum(numeric_only=True).loc['M'].xGA / player_stats.groupby('P').sum(numeric_only=True).loc['M'].MIN)
d_adj = (player_stats.groupby('P').sum(numeric_only=True).loc['D'].xGA / player_stats.groupby('P').sum(numeric_only=True).loc['D'].MIN)
player_stats.loc[player_stats.P == 'F','xGA'] -= f_adj * player_stats.MIN
player_stats.loc[player_stats.P == 'M','xGA'] -= m_adj * player_stats.MIN
player_stats.loc[player_stats.P == 'D','xGA'] -= d_adj * player_stats.MIN

player_stats['xGA'] = (player_stats.GA - player_stats.groupby(['Date','Team']).GA.max().mean() / 90 * player_stats.MIN) * 0.15 + player_stats.xGA * 0.85
player_stats.xGA *= -1

player_stats['xGD'] = player_stats.xGF + player_stats.xGA
player_stats = player_stats[['Date','Type','Team','Player','MIN','xGD','P','In','Out']].rename(columns={'Player':'Name','xGD':'Rtg'})
pd.concat((player_stats,goalie_stats)).to_feather('data/PlayerStats.ftr')