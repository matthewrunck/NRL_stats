# -*- coding: utf-8 -*-
"""
For each of the types of files (ie timeline, player and team stats), the following code 
joins the data in each game file into a single dataframe 
"""

import pandas as pd
import numpy as np
import re
import os


#convert time format mm:ss to float

def get_mins(ser):
    v=ser.str.split(":",expand=True).astype(float).fillna(0)
    mins=v[0]+v[1]/60
    return mins

#Initialize list of string for filenames for player, timeline and team stat files

players=[]
timelines=[]
teams=[]

path=os.getcwd()+'//data'

#Loop through each file in the directory and add each game file to its respective list

for r, d, f in os.walk(path):
    for file in f:
        if "player.csv" in file:
           players.append(os.path.join(r, file))
        if "timeline.csv" in file:
           timelines.append(os.path.join(r, file))
        if "team.csv" in file:
           teams.append(os.path.join(r, file))

#Create a list of dataframes of player stats from each game
player_list = []
for file in players:
    df = pd.read_csv(file).dropna(axis=1)
    df.replace(to_replace='-', value=0, inplace=True)
    df = df.apply(pd.to_numeric,errors='ignore')
    
    #Determine if each player was on winning team
    #find points scored by each team
    sumdf=df.groupby(['Team']).sum()
    
    #team with most points
    winner = sumdf['Points'].idxmax()
    
    df['Won']=winner==df['Team']
    player_list.append(df)


#Join the datframes for each game into a single dataframe
l0 = pd.concat(player_list)

#Strip whitespace from column names
l0.columns = [re.sub(r'\s+', '',c) for c in l0.columns]


l0['AveragePlayTheBallSpeed']=l0['AveragePlayTheBallSpeed'].str.replace('s','').astype(float)
l0['TotalPlayTheBallTime']=l0['AveragePlayTheBallSpeed']*l0['PlayTheBall']
l0['MinsPlayed']=get_mins(l0['MinsPlayed'])
l0['StintOne']=get_mins(l0['StintOne'])
l0['StintTwo']=get_mins(l0['StintTwo'])
l0['MperRun'] = (l0['AllRunMetres']/l0['AllRuns']).fillna(0)
l0.drop(columns="Unnamed:0",inplace=True)

l0 = l0.apply(pd.to_numeric,errors='ignore')

#Two teams have two word names - restore the lost second word
l0.loc[l0['Team']=='Sea','Team']='Sea Eagles'    
l0.loc[l0['Team']=='Wests','Team']='Wests Tigers' 

#save as csv
l0.to_csv('pipeline\\players')
print('players done')

#Create a list of dataframes of timeline stats from each game

timeline_list = []
for file in timelines:
    df = pd.read_csv(file)
    df = df.apply(pd.to_numeric,errors='ignore')
    df['Team'].replace(to_replace='Sea', value='Sea Eagles', inplace=True)    
    df['Team'].replace(to_replace='Wests', value='Wests Tigers', inplace=True)     
    df['Event Type'][df['Event Type'].str.contains('Eagles Interchange')]=df['Event Type'].str.slice(start=7)
    df['Event Type'][df['Event Type'].str.contains('Tigers Interchange')]=df['Event Type'].str.slice(start=7)     
    timeline_list.append(df)

   


l1 = pd.concat(timeline_list)
l1.columns = [re.sub(r'\s+', '',c) for c in l1.columns]
l1['Time']=get_mins(l1['Time'])
l1.drop(columns='Unnamed:0',inplace=True)

print('timelines done')
l1.to_csv('pipeline\\timelines')

#Create a list of dataframes of team stats from each game

team_list =[]
for file in teams:
    df=pd.read_csv(file,dtype=str)
    df['Opponent']=df['Match'].str.split(' vs ',expand=True)[0]
    df['Opponent'][df['Match'].str.split(' vs ',expand=True)[0]==df['Team']]= df['Match'].str.split(' vs ',expand=True)[1]
    team_list.append(df)
    
    
l2 = pd.concat(team_list,sort=False).fillna('0')
l2.columns = [re.sub(r'\s+', '',c) for c in l2.columns]
cols=l2.columns.difference(['Match', 'Team','Opponent'])
l2[cols] = l2[cols].apply(lambda x: x.str.replace('[s,%]',''),axis=0)
l2['Team'][l2['Team']=='Sea']='Sea Eagles'    
l2['Team'][l2['Team']=='Wests']='Wests Tigers' 
l2['TimeInPossession']=get_mins(l2['TimeInPossession'])
l2 = l2.apply(pd.to_numeric,errors='ignore')
a=l2['CompletedSetsRatio'].str.split('/',expand=True).astype(int)
l2['CompleteSets']=a[0]
l2['TotalSets']=a[1]
l2.drop(columns=["Unnamed:0","CompletedSetsRatio"],inplace=True)
l2['CompletedSets(%)']=l2['CompleteSets']/l2['TotalSets']

l2.to_csv('pipeline\\teams')
print('teams done')
