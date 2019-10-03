# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 20:47:25 2019

@author: Lenovo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


surv=pd.read_csv('pipeline\\survival_to_field_goal')
games = pd.read_csv('pipeline\\games')
year=2019
match = 'Sharks vs Raiders'
game=games[(games['Year']==year)&(games['Match']==match)]

#This function returns a non-parametric survival function for time until a fieldgoal is scored
#for a game with a particular margin at a particular time

def nonparsurv(margin,time):
    #returns all instances with a given margin at thhe given time
    m=surv.loc[(np.abs(surv['Margin'])==abs(margin))&(surv['Start']<=time*60)&(surv['Stop']>time*60)]
    #if no instances, then return values of 100,0
    if m.shape[0]==0:
        return 100,0
    else:
        #if either team kicked a field goal
        m['Field']=m['Field_for'] | m['Field_ag']
        #take the given time as the starting point
        m['Start']=time*60
        #duration from given time to next scoring event
        m['Time']=m['Stop']-m['Start']
        #percentage of field goals kicked by leading team
        f=np.sum(m['Field_for'])/np.sum(m['Field'])
        #if scores level, assume either team has equal probability of kicking goal
        if margin==0:
            f=0.5
    
        #Sort by duration to next score
        m.sort_values(by='Time',inplace=True)
        #returns the 'risk set'- the number of times with no score where duration was as great or greater
        m['Risk']= m.shape[0]-np.arange(m.shape[0],)
        #Given no score until now,risk the next score will be field goal
        m['Cond_no_goal']=(m['Risk']-m['Field'])/m['Risk']
        #probbilty of no field goal occurring to this duration is cumulative product of conditional probabilities
        m['SurvProb']=np.cumprod(m['Cond_no_goal'])
        
        #make a random sample for the tme of the next field goal
        u=np.random.uniform()
        #if we dont expect a field goal to be kicked, we say the nnext expected field goal is in 100 minutes
        if u < np.min(m['SurvProb']):
            t=100
        #otherwise it is the time when the survival probabilty is u
        else:
            t=m[m['SurvProb']<=u]['Time'].iloc[0]/60
        
        return t,f



round_avs = games.groupby(['Year','Round']).mean()['tries_scored_per_game']

plt.plot(np.arange(len(round_avs)),round_avs)

#simulate match



def sim_match(game,time=0,home_score=0,away_score=0):
    t=time
    r = game['Round'].iloc[0]
    round_av = round_avs.loc[game['Year'].iloc[0],r]
    home = game[game['Home']]
    away = game[game['Home']==False]
    #expected number of home tries and penalty goals
    home_tries_mean =home['tries_scored_per_game'].iloc[0]*away['tries_conceded_per_game'].iloc[0]/round_av
    home_pens_mean =  home['penalty_goals_per_game'].iloc[0]
    #expected number of away team tries and penalty goals
    away_tries_mean = away['tries_scored_per_game'].iloc[0]*home['tries_conceded_per_game'].iloc[0]/round_av
    away_pens_mean =  away['penalty_goals_per_game'].iloc[0]
    
    #conversion accuracy of home team
    home_conv_acc = (home['conversions_per_game']/home['tries_scored_per_game']).iloc[0]
    
    #conversion accuracy of away team
    away_conv_acc = (away['conversions_per_game']/away['tries_scored_per_game']).iloc[0]
    
    #time spent not playing due to stoppages after scoring
    time_off = home_tries_mean + home_pens_mean + away_tries_mean + away_pens_mean

    
    match_score = pd.Series({'Home':home_score,'Away':away_score})
    
    home_sum = {'Try':[],'Conversion':[],'Penalty Goal':[],'Field Goal':[]}
    away_sum = {'Try':[],'Conversion':[],'Penalty Goal':[],'Field Goal':[]}
    
    
    while t <80:

        margin=match_score['Home']-match_score['Away']
        #home and away field goal times initialised to 100 mins
        hf=100
        af=100
        #get time of next field goal 
        t0,p=nonparsurv(margin,t)
        if t0<100:
            #if there is a field goal, deciding whether it is kicked by home or away team
            u=np.random.uniform()
            if ((margin<=0)&(u>p))|((margin>0)&(u<p)):
                hf=t0
            else:
                af=t0
        
        #Expected time until next home try ( adjusted to account for time off)
        mu0_t = (80-time_off)/home_tries_mean
        mu0_p = (80-time_off)/home_pens_mean
    
        mu1_t = (80-time_off)/away_tries_mean
        mu1_p = (80-time_off)/away_pens_mean
        
        
        #Get a random sample for time to each possible scoring event
        next_score = {'home_try':np.random.exponential(mu0_t),'home_pen':np.random.exponential(mu0_p),'home_field':hf,'away_try':np.random.exponential(mu1_t),'away_pen':np.random.exponential(mu1_p),'away_field':af}
        #The type of scoring event with the smallest time is considered to be the next score 
        score = min(next_score, key=next_score.get)
        
        t+=next_score[score]
        
        if t>80:
            break
        #Update score according to type of score
        if score == 'home_try':
            conv = np.random.uniform()<home_conv_acc
            match_score.loc['Home']+=4+2*conv
            home_sum['Try'].append(t)
            if conv:
                home_sum['Conversion'].append(t)
            
        elif score == 'away_try':
            conv = np.random.uniform()<away_conv_acc
            match_score.loc['Away']+=4+2*conv
            away_sum['Try'].append(t)
            if conv:
                away_sum['Conversion'].append(t)
        
        elif score == 'home_field':
            match_score.loc['Home']+=1
            home_sum['Field Goal'].append(t)
        
        elif score == 'away_field':
            match_score.loc['Away']+=1
            away_sum['Field Goal'].append(t)
        
        elif score =='home_pen':
            match_score['Home']+=2
            home_sum['Penalty Goal'].append(t)
            
        else:
            match_score['Away']+=2
            away_sum['Penalty Goal'].append(t)
            
        t+=1
    return match_score, home_sum, away_sum
    
    
    
final_score,home,away = sim_match(game,time=0)

print(final_score)
print(home)
print(away)


#Now we repeat the simulate to get the distribution f possible outcomes
def MC_sim(n,game,time=0,home_score=0,away_score=0):
    print(game['Match'].iloc[0]+' ' + str(game['Year'].iloc[0]))
    x=np.zeros((3,n))
    
    for i in np.arange(n):
        x[:2,i],home,away=sim_match(game,time=time,home_score=home_score,away_score=away_score)
        x[2,i] = len(home['Field Goal'])+len(away['Field Goal'])
        if (i%100==0)&(i>0):
            print('simulation ' + str(100*i/n) + '% complete')
    return x
        
y=MC_sim(5000,game,time=60,home_score=12,away_score=12)

fg = np.mod(y,2)
np.sum(fg)

r_win =np.mean(y[0,:]<y[1,:])
r_tie = np.mean(y[0,:]==y[1,:])
r_lose=1-(r_win+r_tie)  

fig=plt.figure() #set up the figures
fig.set_size_inches(6, 5)
ax=fig.add_subplot(1,1,1)
plt.bar([1,2,3],[r_win,r_tie,r_lose],tick_label=['Raiders Win','Draw','Sharks Win'],color =[team_cols['Raiders'],'yellow',team_cols['Sharks']] )
plt.title('Probability of outcomes at 60th minute (based on 5000 simulations)') 
plt.ylabel('Probability')

fig.savefig('images\\outcomes_min_60.png')

y1=MC_sim(5000,game,time=60,home_score=12,away_score=13)



r1_win =np.mean(y1[0,:]<y1[1,:])
r1_tie = np.mean(y1[0,:]==y1[1,:])
r1_lose=1-(r1_win+r1_tie)  

fig=plt.figure() #set up the figures
fig.set_size_inches(6, 5)
ax=fig.add_subplot(1,1,1)
plt.bar([1,2,3],[r1_win,r1_tie,r1_lose],tick_label=['Raiders Win','Draw','Sharks Win'],color =[team_cols['Raiders'],'yellow',team_cols['Sharks']] )
plt.title('Probability of outcomes after field goal (based on 5000 simulations)') 
plt.ylabel('Probability')

fig.savefig('images\\outcomes_min_60_after field.png')

y2=MC_sim(5000,game,time=60,home_score=12,away_score=16)



r2_win =np.mean(y2[0,:]<y2[1,:])
r2_tie = np.mean(y2[0,:]==y2[1,:])
r2_lose=1-(r2_win+r2_tie)  

y3=MC_sim(5000,game,time=60,home_score=12,away_score=18)



r3_win =np.mean(y3[0,:]<y3[1,:])
r3_tie = np.mean(y3[0,:]==y3[1,:])
r3_lose=1-(r3_win+r3_tie)  



con = game.loc[game['Home']==False,'conversions_per_game'].iloc[0]/game.loc[game['Home']==False,'tries_scored_per_game'].iloc[0]

r_try_win = r2_win*(1-con) +r3_win*con
r_try_tie = r2_tie*(1-con) +r3_tie*con
r_try_lose = r2_lose*(1-con) +r3_lose*con
  


fig=plt.figure() #set up the figures
fig.set_size_inches(6, 5)
ax=fig.add_subplot(1,1,1)
plt.bar([1,2,3],[r_try_win,r_try_tie,r_try_lose],tick_label=['Raiders Win','Draw','Sharks Win'],color =[team_cols['Raiders'],'yellow',team_cols['Sharks']] )
plt.title('Probability of outcomes after try (based on 5000 simulations)') 
plt.ylabel('Probability')

fig.savefig('images\\outcomes_min_60_after_try.png')
  
plt.hist(y[1,:]-y[0,:])
plt.hist(y[1,:])

fig,axes=plt.subplots(1,3)
axes=[ax1,ax2,ax3]

#Test to see if we get a reasonable number of field goals

def yep(game):
    y=MC_sim(10,game,time=0,home_score=0,away_score=0)
    return np.sum(y[2,:])

g=games.groupby(['Year','Match'],as_index=False).apply(lambda x:yep(x))

g.to_csv('pipeline\\goal_sims')

def outcomes(game,time,home=True,home_score=0,away_score=0,reps=1000):
    
    if_goal = MC_sim(reps,game,time=time+1,home_score=home_score+1*home,away_score=away_score+1*(1-home))
    if_try_conv =  MC_sim(reps,game,time=time+1,home_score=home_score+6*home,away_score=away_score+6*(1-home))
    if_try_no_conv = MC_sim(reps,game,time=time+1,home_score=home_score+4*home,away_score=away_score+4*(1-home))
    if_no_score = MC_sim(reps,game,time=time,home_score=home_score,away_score=away_score)
    
    
    results = {'if_goal':if_goal,'if_try_conv':if_try_conv,'if_try_no_conv':if_try_no_conv,'if_no_score':if_no_score}
    
    win_prob_if_goal = np.mean(if_goal[1-home,:]>if_goal[home*1,:])
    win_prob_if_try_conv = np.mean(if_try_conv[1-home,:]>if_try_conv[home*1,:])
    win_prob_if_try_no_conv = np.mean(if_try_no_conv[1-home,:]>if_try_no_conv[home*1,:])
    win_prob_if_no_score = np.mean(if_no_score[1-home,:]>if_no_score[home*1,:])

    tie_prob_if_goal = np.mean(if_goal[1-home,:]==if_goal[home*1,:])
    tie_prob_if_try_conv = np.mean(if_try_conv[1-home,:]==if_try_conv[home*1,:])
    tie_prob_if_try_no_conv = np.mean(if_try_no_conv[1-home,:]==if_try_no_conv[home*1,:])
    tie_prob_if_no_score = np.mean(if_no_score[1-home,:]==if_no_score[home*1,:])
    
    conv_prob=game['conversions_per_game'].iloc[1-home]/game['tries_scored_per_game'].iloc[1-home]
    
    outcome_probs = {'conv_prob':conv_prob,'win_prob_if_goal':win_prob_if_goal,'win_prob_if_try_conv':win_prob_if_try_conv,'win_prob_if_try_no_conv':win_prob_if_try_no_conv,'win_prob_if_no_score':win_prob_if_no_score,
                     'tie_prob_if_goal':tie_prob_if_goal,'tie_prob_if_try_conv':tie_prob_if_try_conv,'tie_prob_if_try_no_conv':tie_prob_if_try_no_conv,'tie_prob_if_no_score':tie_prob_if_no_score}
    
    return outcome_probs,results


a,b= outcomes(game,time=70,home=False,home_score=12,away_score=12,reps=100)

def decider(outcome_probs, field_prob, try_prob):
    win_prob_if_attempt = field_prob*outcome_probs['win_prob_if_goal'] + (1-field_prob)*outcome_probs['win_prob_if_no_score']
    win_prob_if_no_attempt = try_prob*(outcome_probs['conv_prob']*outcome_probs['win_prob_if_try_conv']+(1-conv_prob)*outcome_probs['win_prob_if_try_no_conv'])+(1-try_prob)*outcome_probs['win_prob_if_no_score']  
    tie_prob_if_attempt = field_prob*tie_prob_if_goal + (1-field_prob)*outcome_probs['tie_prob_if_no_score']
    tie_prob_if_no_attempt = try_prob*(conv_prob*outcome_probs['tie_prob_if_try_conv']+(1-conv_prob)*outcome_probs['tie_prob_if_try_no_conv'])+(1-try_prob)*outcome_probs['tie_prob_if_no_score']


outcome_prob_lists = []
for i in np.arange(60,80):
    outs, b = outcomes(game,time=i,home=False,home_score=12,away_score=12,reps=1000)
    outcome_prob_lists.append(outs)
    print(i)
    
outcomes = pd.DataFrame(outcome_prob_lists)

wins = np.array(outcomes.iloc[:,1:5])+np.array(outcomes.iloc[:,5:])*0.5

w=np.zeros((20,3))
w[:,0] = wins[:,0]
w[:,1] = wins[:,1]*outcomes.iloc[0,0]+wins[:,2]*(1-outcomes.iloc[0,0])
w[:,2] = wins[:,3]

np.savetxt('pipeline\\raiders_win',w,delimiter=",")

plt.plot(np.arange(60,80),w)
plt.legend(['Win field goal','Win try','Win no score'])


plt.plot(wins[:,3])
