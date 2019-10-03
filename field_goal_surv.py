# -*- coding: utf-8 -*-
"""
The following code takes the match timelines and for each match creates a record for the score for each team at each second of the match
These scorelines are then used to create a dataframe survival analysis, where every score change is treated as an event.
For each score change the time of the current score, the time of the current score, the margin at the time of the score, whether the score
change was the result of a field goal is recorded
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

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

games= pd.read_csv('pipeline\\games')
with open('pipeline\\match_timelines.json') as data_file:
    data = json.load(data_file)
match_timelines = pd.DataFrame.from_dict(data)

def get_scoreline(match,year):
    #1-d array containg each second in the match
    match_time=np.linspace(0,80,4801)
    #Get the round of the match-only really want ths for labelling the match plot
    Round = games.loc[(games['Match']==match)].loc[(games['Year']==year),'Round'].iloc[0] 
    out_dict = {'Time':list(match_time)}   
    scores= {'Try':4,'Conversion-Made':2,'Penalty Shot-Made':2,'Drop Goal-Made':1}
    #get the timelines of the particular match for eah team
    match_events = match_timelines.loc[(match_timelines['Match']==match)&(match_timelines['Year']==year)]
    for team in match_events.Team.unique():
        #the match_timelines dataframe contains three rows per match- we ignore the row for events not belonging to a particular team
        if team == '':
            continue
        else:
            team_events=match_events.loc[match_events['Team']==team]
            #score for team at each second
            team_score =  np.zeros(4801)
            for score in scores:
                if team_events[score].isna().all():
                    print(score)
                    continue
                else:
                    for x in team_events[score].iloc[0]:
                        team_score=np.where((match_time<x),team_score,team_score+scores[score]) 
        out_dict[team]=list(team_score)
        
    scoreline=pd.DataFrame(out_dict)

    return scoreline




#Function to apply get_scoreline to every match
def get_all_scorelines(group):   
    match = group['Match'].iloc[0]
    year = group['Year'].iloc[0]
    print(match+str(year))
    try:
        score = [get_scoreline(match,year)]
        return score
    except:
        pass
        return None


#returns a series of datframes containing the scores for each team at every second of the match
all_scorelines = match_timelines.groupby(['Match','Year'],as_index=False).apply(lambda x: get_all_scorelines(x))

#there are a couple matches on NRL.com with empty timelines- lets drop these
all_scorelines = [item for item in list(all_scorelines.dropna().apply(lambda x:x[0])) if item.shape[1]==3]

#convert the series to a 3-dimensional numpy array
all_scoreline=np.stack(all_scorelines)


#change of margin
margins=(all_scoreline[:,:,1]-all_scoreline[:,:,2]).T

np.save('pipeline\\margins')

mar = np.unique(margins,return_counts=True)
plt.bar(mar[0],mar[1])

"""

"""

mat = []
start = []
stop = []
mar = []
for_event = []
ag_event= []
#Loop through each match
for i in np.arange(margins.shape[1]):
    #Assign match index to event
    mat.append(i)
    #For each new match initialize new event
    start.append(0)
    #Loop though each second of match
    for j in np.arange(1,margins.shape[0]):
        #Find when score changes
        if margins[j,i]!=margins[j-1,i]:
            #Record time score changes
            stop.append(j)
            mat.append(i)
            if abs(margins[j,i])-abs(margins[j-1,i])==1:
                #If size of margin increase by 1, record field goal for leading team
                for_event.append(True)
                ag_event.append(False)
            elif abs(margins[j,i])-abs(margins[j-1,i])==-1:
                #If size of margin decreases by 1, record field goal against leading team
                for_event.append(False)
                ag_event.append(True)
            else:
                #Any other score change -  no field goal
                for_event.append(False)
                ag_event.append(False)
            if j <margins.shape[0]:
                #If the score happens before full time, start another event
                start.append(j)
                #margin is margin just before score
                mar.append(margins[j-1,i])
        if j==margins.shape[0]-1:
            #at full time (80 mins), end current event 
            stop.append(margins.shape[0])
            for_event.append(False)
            ag_event.append(False)
            mar.append(margins[-1,i])
            
 #Convert list of dicts to dataframe           
long_form = pd.DataFrame({'ID':mat,'Start':start,'Stop':stop,'Margin':mar,'Field_for':for_event,'Field_ag':ag_event})                
    

long_form.to_csv('pipeline\\survival_to_field_goal')

def nonparsurv(margin,time):
    m=long_form.loc[(np.abs(long_form['Margin'])==abs(margin))&(long_form['Start']<=time*60)&(long_form['Stop']>time*60)]
    if m.shape[0]==0:
        return 100,0
    else:
        m['Field']=m['Field_for'] | m['Field_ag']
        m['Start']=time*60
        m['Time']=m['Stop']-m['Start']
        f=np.sum(m['Field_for'])/np.sum(m['Field'])
        if margin==0:
            f=0.5
    
        m.sort_values(by='Time',inplace=True)
        m['Risk']= m.shape[0]-np.arange(m.shape[0],)
        m['Cond_no_goal']=(m['Risk']-m['Field'])/m['Risk']
        m['SurvProb']=np.cumprod(m['Cond_no_goal'])
        u=np.random.uniform()
        if u < np.min(m['SurvProb']):
            t=100
        else:
            t=m[m['SurvProb']<=u]['Time'].iloc[0]/60
        
        return t,f


