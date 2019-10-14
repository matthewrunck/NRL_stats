

import seaborn as sns
import pandas as pd


players=pd.read_csv('pipeline\\players')
team=pd.read_csv('pipeline\\teams')
timelines = pd.read_csv('pipeline\\timelines')





"""
Our 'timelines' dataframe consists of a row for each event that occurs, containing columns for the event type, the match and year
the event happened in, the player involved, as well as the game time at which the event occurred.

We want a dataframe where each row contains information for each of the event that occurred during a particular match.

Each row is the events happening to a particular team during a particular match.
Each column corresponds to particular event type, say a try being scored, or an error occurring.
Each data value is a list of times at which that particular event occurred during that particular match to that particular team 

First we obtain a list of series for each team in each match, then convert to a dataframe
"""

new_cols =  ['Match','Player','Team','Year']
new_cols=new_cols+timelines['EventType'].unique().tolist()


list_timelines=[]
for year in timelines['Year'].unique():
    #loop through the years
    df0=timelines[timelines['Year']==year].fillna('')
    for match in df0['Match'].unique():
        #Loop through the matches in that year
        df1=df0[df0['Match']==match].fillna('')
        for team in df1['Team'].unique():
            #Loop through teams in that match
            df2=df1[df1['Team']==team].fillna('')
            
            ser=pd.Series(index=new_cols)
            ser['Year'] = year
            ser['Match']=match
            ser['Team'] = team
            
            for event in df2['EventType'].unique():
                #For each unique event type in the match, get a list of times the event occurred
                df = df2[df2['EventType']==event]    
                ser[event] = df['Time'].tolist()
            list_timelines.append(ser)
    print(year)






match_timelines= pd.DataFrame(list_timelines)
with open('pipeline\\match_timelines.json', 'w') as f:
    json.dump(match_timelines.to_dict(), f)





#returns a series of all the times an event has occurred in all matches
def all_times(event):
    a=match_timelines[event].apply(pd.Series).stack().reset_index(drop=True)
    return a

#See the distribution of scoring events
a=all_times('Drop Goal-Made')
plt.hist(a,bins=np.arange(0,80,4))

b=all_times('Penalty Shot-Made')
plt.hist(b,bins=np.arange(0,80,2))

c=all_times('Try')
plt.hist(c,bins=np.arange(0,80,1))



#This function tells us the last time a particular event occurred
def get_max(x):
    try:
        if max(x)<=80:
            return max(x)
        else:
            x=pd.Series(x)
            return np.max(x[x<=80])
    except:
        return -1
#When are the last points scored
e=match_timelines['Try'].map(lambda x: get_max(x))
f=match_timelines['Penalty Shot-Made'].map(lambda x: get_max(x))
g=match_timelines['Drop Goal-Made'].map(lambda x: get_max(x))
last_points =pd.concat([e,f,g],axis=1).max(axis=1)
last_points=last_points[last_points>0]
match_timelines['final_score']=last_points
plt.hist(last_points,bins=80)
np.median(last_points)








