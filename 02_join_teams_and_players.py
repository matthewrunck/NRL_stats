# -*- coding: utf-8 -*-
"""
Merge the team and player stats to get aggregated stats for each game, including oppositions stats.
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

#Associating a colour with ech team
team_cols = {'Storm':'#011641',
             'Rabbitohs':'#003C1A',
             'Roosters':'#000080',
             'Raiders':'#32CD32',
             'Sea Eagles':'#6F0F3B',
             'Eels':'#006EB5',
             'Panthers':'#221F20',
             'Broncos':'#FABF16',
             'Knights':'#EE3524',
             'Wests Tigers':'#F68C1A',
             'Sharks':'#00A9D8',
             'Warriors':'#BDBCBC',
             'Cowboys':'#FFDD02',
             'Dragons':'#E2231B',
             'Bulldogs':'#00539F',
             'Titans':'#FBB03F'}


players=pd.read_csv('pipeline\\players')
players=players.apply(pd.to_numeric,errors='ignore')
team=pd.read_csv('pipeline\\teams').apply(pd.to_numeric,errors='ignore')
timelines = pd.read_csv('pipeline\\timelines')

#Add the stats for each player on the same team in the same match to get aggregate stats for the team
games_for=players.groupby(['Year','Match','Team']).sum()
#Keep the unaggrgated stat for the Round
games_for['Round']=players.groupby(['Year','Match','Team']).first()['Round']
#Keep the grouping values ('Year', 'Match' and 'Team') as columns
games_for=games_for.reset_index()

cols=['Year','Match','Team']+list(set(games_for.columns)-set(team.columns))

#join the team stats to the aggregated player stats
games_for=team.merge(games_for[cols],left_on=['Year','Match','Team'],right_on=['Year','Match','Team'])

#We add the opponents stats to each teams  game row - ie for each stat we have a for stat '+' and an against'-'
games= games_for.merge(games_for,how='left', left_on=['Year','Match','Team'], right_on = ['Year','Match','Opponent'],suffixes=('+', '-'))

#Making some adjustment- adding game result, whether team was home team, adjusting some stats that 
#are averages rather than sums

games['Won']=(games['Points+']>games['Points-'])*1
games['Draw']=(games['Points+']==games['Points-'])*1
games['MperRun+']=games['AllRunMetres+']/games['AllRuns+']
games['MperRun-']=games['AllRunMetres-']/games['AllRuns-']
games['MatchesPlayed']=1
games['Round']=games['Round+'].astype(int, errors='ignore')
games['Home']=(games['Match'].str.split(expand=True).iloc[:,0]==games['Team+'].str.split(expand=True).iloc[:,0])
games.rename(columns={'Team+':'Team','Team-':'Opp'},inplace=True)
games.drop(columns=['Number+','MinsPlayed+','Number-','MinsPlayed-','Won-','Round+','Round-'],inplace=True)

#This function return the average of a stat from the teams previous n_games games,
#eg average number of tries scored per game over the previous 20 games

def av_events_per_game(group,event,n_games):
    #Ordering the games in chronologincal order
    a=games.sort_values(by=['Year','Round'])
    #Average of all teams over all games
    tot_ave=np.mean(a[event])
    
    #The ordered games of the team of interest
    b=a[(a['Team']==group['Team'].iloc[0])&(a['Home']==group['Home'].iloc[0])].reset_index()

        
    #c is the match of interest, and ix is its index in the ordered set
    c=b[(b['Match']==group['Match'].iloc[0])&(b['Year']==group['Year'].iloc[0])]
    ix=c.index.item()
    
    #if there are less than n_games previous games, we use a weighted average between the total average and the teams 
    #average up to that point. After n_games we take the average of the previous n_games games
    if ix < n_games:
        
        if ix==0:
            d=tot_ave
        else:
            d=(tot_ave*(n_games-ix) + np.sum(b[event].iloc[:c.index.item()]))/24
    else:
        d=np.mean(b[event].iloc[c.index.item()-int(n_games/2):c.index.item()])
    return d

#apply this fnction across the rows of games to obtain rolling averages for tries scored and conceded, penalties scored and conversions made
tries_scored_per_game = games.groupby(['Year','Match','Team'],as_index=False).apply(lambda x:av_events_per_game(x,'Tries+',24))
games['tries_scored_per_game'] = games.merge(tries_scored_per_game.reset_index(),on=['Year','Match','Team']).iloc[:,-1]

tries_conceded_per_game = games.groupby(['Year','Match','Team'],as_index=False).apply(lambda x:av_events_per_game(x,'Tries-',24))
games['tries_conceded_per_game'] = games.merge(tries_conceded_per_game.reset_index(),on=['Year','Match','Team']).iloc[:,-1]

penalty_goals_per_game = games.groupby(['Year','Match','Team'],as_index=False).apply(lambda x:av_events_per_game(x,'PenaltyGoals+',24))
games['penalty_goals_per_game'] = games.merge(penalty_goals_per_game.reset_index(),on=['Year','Match','Team']).iloc[:,-1]


conversions_per_game = games.groupby(['Year','Match','Team'],as_index=False).apply(lambda x:av_events_per_game(x,'Conversions+',24))
games['conversions_per_game'] = games.merge(conversions_per_game.reset_index(),on=['Year','Match','Team']).iloc[:,-1]

games.to_csv('pipeline\\games')


