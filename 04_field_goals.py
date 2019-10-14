# -*- coding: utf-8 -*-
"""
The following code produces dataframes with information for all field goal attempts including the score at the time of the field goal and the final score
"""

with open('pipeline\\match_timelines.json') as data_file:
    data = json.load(data_file)
match_timelines = pd.DataFrame.from_dict(data)
#Find a team's score in a given match at a given time
def get_score(team,match,year,time):
    m=match_timelines[(match_timelines['Team']==team)&(match_timelines['Match']==match)&(match_timelines['Year']==year)]    
    tries=np.sum(pd.Series(m['Try'].tolist()[0])<time)
    conversions = np.sum(pd.Series(m['Conversion-Made'].tolist()[0])<time)
    penalties = np.sum(pd.Series(m['Penalty Shot-Made'].tolist()[0])<time)
    dropgoals = np.sum(pd.Series(m['Drop Goal-Made'].tolist()[0])<time)
    score = tries*4+conversions*2+penalties*2+dropgoals
    return score



#For ever field goal attempt get score at the time it was attempted, as well as score at 80 minutes
def score_at_field_goal(group,success=True):
    if success==True:
        x='Drop Goal-Made'
    else:
        x='Drop Goal-Missed'
    if group[x].isnull().all():
        return None
    else:
        goals=[]
        scores_at_goal_for=[]
        scores_at_goal_against=[]
        final_scores_for=[]
        final_scores_against=[]
        matches = []
        years = []
        for team in group['Team']:
            if group[group['Team']==team][x].notna().any():
                opp = list(set(group['Team'])-set([team,'']))[0]
                for goal in group[group['Team']==team][x].iloc[0]:
                    if goal>80:
                        continue
                    time=goal
                    match = group['Match'].iloc[0]
                    year = group['Year'].iloc[0]
                    print(str(match)+str(year)+str(time))
                    score_at_goal_for = [get_score(team,match,year,time)]
                    score_at_goal_against = [get_score(opp,match, year,time)]
                    final_score_for = [get_score(team,match,year,80.01)]
                    final_score_against =  [get_score(opp,match,year,80.01)]
                    matches += [match]
                    years += [year]
                    goals += [time]
                    scores_at_goal_for += score_at_goal_for
                    scores_at_goal_against += score_at_goal_against
                    final_scores_for += final_score_for
                    final_scores_against += final_score_against
        return {'Year':years,'Match':matches,'Time':goals,'Score at Goal For':scores_at_goal_for,
                'Score at Goal Against':scores_at_goal_against,'Final Score For':final_scores_for,'Final Score Against':final_scores_against}

#apply function to get scores for all successful field goal attempts
grouped_made=match_timelines.groupby(['Year','Match'],as_index=False).apply(lambda x:score_at_field_goal(x,success=True))


#Change the series obtained from appling function to list and then back to dataframe
list_field_made = []
for group in grouped_made[grouped_made.notnull()]:
    list_field_made.append(pd.DataFrame(group))
    
field_made = pd.concat(list_field_made)

#include margins and match result
field_made['Margin at Goal'] = field_made['Score at Goal For']  - field_made['Score at Goal Against']    
field_made['Final Margin']=field_made['Final Score For'] - field_made['Final Score Against']
field_made['Result']='Won'
field_made.loc[field_made['Final Margin']<0,'Result']='Lost'
field_made.loc[field_made['Final Margin']==0,'Result']='Tied'
field_made['points_against_after_made'] = field_made['Final Score Against']- field_made['Score at Goal Against']

field_made.to_csv('pipeline\\field_made')

plt.hist([margins.flatten(),field_made['Margin at Goal']],normed=True,bins=np.arange(-20.5,22.5,1))



margin_at_goal_plot=sns.pairplot(x_vars=['Time'],y_vars=['Margin at Goal'], data=field_made,hue='Result',size=4,plot_kws={"s": 20})
margin_at_goal_plot.savefig('images\\margin_at_goal.png')


points_after_goal = sns.pairplot(x_vars=['Time'],y_vars=['points_against_after_made'], data=field_made,hue='Result',size=4,plot_kws={"s": 20})
points_after_goal.savefig('images\\points_after_goal.png')



a=field_made[field_made['Margin at Goal']==0]
b=np.mean(field_made['Result']=='Won')
c=np.mean(field_made['Result']=='Tied')



#Repeat the process for missed field goal attempts
grouped_missed=match_timelines.groupby(['Year','Match'],as_index=False).apply(lambda x:score_at_field_goal(x,success=False))

list_field_missed = []
for group in grouped_missed[grouped_missed.notnull()]:
    list_field_missed.append(pd.DataFrame(group))
    
field_missed = pd.concat(list_field_missed)
field_missed['Margin at Goal'] = field_missed['Score at Goal For']  - field_missed['Score at Goal Against']    
field_missed['Final Margin']=field_missed['Final Score For'] - field_missed['Final Score Against']
field_missed['Result']='Won'
field_missed.loc[field_missed['Final Margin']<0,'Result']='Lost'
field_missed.loc[field_missed['Final Margin']==0,'Result']='Tied'
field_missed['points_against_after_missed'] = field_missed['Final Score Against']- field_missed['Score at Goal Against']

field_missed.to_csv('pipeline\\field_missed')

sns.pairplot(x_vars=['Time'],y_vars=['Margin at Goal'], hue='Result',data=field_missed,size=4,plot_kws={"s": 20})
sns.pairplot(x_vars=['Time'],y_vars=['points_against_after_missed'], hue='Result',data=field_missed,size=4,plot_kws={"s": 20})


def score_at_penalty_goal(group):
    x='Penalty Shot-Made'    
    if group[x].isnull().all():
        return None
    else:
        goals=[]
        scores_at_goal_for=[]
        scores_at_goal_against=[]
        final_scores_for=[]
        final_scores_against=[]
        matches = []
        years = []
        for team in group['Team']:
            if group[group['Team']==team][x].notna().any():
                opp = list(set(group['Team'])-set([team,'']))[0]
                for goal in group[group['Team']==team][x].iloc[0]:
                    if goal>80:
                        continue
                    time=goal
                    match = group['Match'].iloc[0]
                    year = group['Year'].iloc[0]
                    print(str(match)+str(year)+str(time))
                    score_at_goal_for = [get_score(team,match,year,time)]
                    score_at_goal_against = [get_score(opp,match, year,time)]
                    final_score_for = [get_score(team,match,year,80.01)]
                    final_score_against =  [get_score(opp,match,year,80.01)]
                    matches += [match]
                    years += [year]
                    goals += [time]
                    scores_at_goal_for += score_at_goal_for
                    scores_at_goal_against += score_at_goal_against
                    final_scores_for += final_score_for
                    final_scores_against += final_score_against
        return {'Year':years,'Match':matches,'Time':goals,'Score at Goal For':scores_at_goal_for,
                'Score at Goal Against':scores_at_goal_against,'Final Score For':final_scores_for,'Final Score Against':final_scores_against}

#apply function to get scores for all successful field goal attempts
grouped_pen_made=match_timelines.groupby(['Year','Match'],as_index=False).apply(lambda x:score_at_penalty_goal(x))


#Change the series obtained from appling function to list and then back to dataframe
list_pen_made = []
for group in grouped_pen_made[grouped_pen_made.notnull()]:
    list_pen_made.append(pd.DataFrame(group))
    
pen_made = pd.concat(list_pen_made)

#include margins and match result
pen_made['Margin at Goal'] = pen_made['Score at Goal For']  - pen_made['Score at Goal Against']    
pen_made['Final Margin']=pen_made['Final Score For'] - pen_made['Final Score Against']
pen_made['Result']='Won'
pen_made.loc[pen_made['Final Margin']<0,'Result']='Lost'
pen_made.loc[pen_made['Final Margin']==0,'Result']='Tied'
pen_made['points_against_after_made'] = pen_made['Final Score Against']- pen_made['Score at Goal Against']

pen_made.to_csv('pipeline\\pen_made')

plt.plot('Time','Margin at Goal','.',data=pen_made)
pen_mar=np.unique(pen_made['Margin at Goal'],return_counts=True)
plt.bar(pen_mar[0],pen_mar[1]/sum(pen_mar[1]))
plt.bar(mar[0],mar[1]/np.sum(mar[1]))
margins=np.load('pipeline\\margins.npy')
plt.hist([margins.flatten(),pen_made['Margin at Goal']],normed=True,bins=np.arange(-20.5,22.5,1))




def score_at_try(group):
    x='Try'    
    if group[x].isnull().all():
        return None
    else:
        goals=[]
        scores_at_goal_for=[]
        scores_at_goal_against=[]
        final_scores_for=[]
        final_scores_against=[]
        matches = []
        years = []
        for team in group['Team']:
            if group[group['Team']==team][x].notna().any():
                opp = list(set(group['Team'])-set([team,'']))[0]
                for goal in group[group['Team']==team][x].iloc[0]:
                    if goal>80:
                        continue
                    time=goal
                    match = group['Match'].iloc[0]
                    year = group['Year'].iloc[0]
                    print(str(match)+str(year)+str(time))
                    score_at_goal_for = [get_score(team,match,year,time)]
                    score_at_goal_against = [get_score(opp,match, year,time)]
                    final_score_for = [get_score(team,match,year,80.01)]
                    final_score_against =  [get_score(opp,match,year,80.01)]
                    matches += [match]
                    years += [year]
                    goals += [time]
                    scores_at_goal_for += score_at_goal_for
                    scores_at_goal_against += score_at_goal_against
                    final_scores_for += final_score_for
                    final_scores_against += final_score_against
        return {'Year':years,'Match':matches,'Time':goals,'Score at Goal For':scores_at_goal_for,
                'Score at Goal Against':scores_at_goal_against,'Final Score For':final_scores_for,'Final Score Against':final_scores_against}

#apply function to get scores for all successful field goal attempts
grouped_try_made=match_timelines.groupby(['Year','Match'],as_index=False).apply(lambda x:score_at_try(x))


#Change the series obtained from appling function to list and then back to dataframe
list_try_made = []
for group in grouped_try_made[grouped_try_made.notnull()]:
    list_try_made.append(pd.DataFrame(group))
    
try_made = pd.concat(list_try_made)

#include margins and match result
try_made['Margin at Goal'] = try_made['Score at Goal For']  - try_made['Score at Goal Against']    
try_made['Final Margin']=try_made['Final Score For'] - try_made['Final Score Against']
try_made['Result']='Won'
try_made.loc[try_made['Final Margin']<0,'Result']='Lost'
try_made.loc[try_made['Final Margin']==0,'Result']='Tied'
try_made['points_against_after_made'] = try_made['Final Score Against']- try_made['Score at Goal Against']

try_made.to_csv('pipeline\\try_made')

plt.plot('Time','Margin at Goal','.',data=try_made)
try_mar=np.unique(try_made['Margin at Goal'],return_counts=True)
plt.bar(try_mar[0],try_mar[1]/sum(try_mar[1]))
plt.bar(mar[0],mar[1]/np.sum(mar[1]))

plt.hist([margins.flatten(),try_made['Margin at Goal']],normed=True,bins=np.arange(-20.5,22.5,1))
