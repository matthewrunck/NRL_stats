# -*- coding: utf-8 -*-
"""
Generting plots
"""



import pandas as pd
import numpy as np
import seaborn as sns
import json
import matplotlib.patches as pat
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import skellam
 
plt.clf()

games= pd.read_csv('pipeline\\games')
with open('pipeline\\match_timelines.json') as data_file:
    data = json.load(data_file)
match_timelines = pd.DataFrame.from_dict(data)

year=2019
match = 'Sharks vs Raiders'
game=games[(games['Year']==year)&(games['Match']==match)]



#Associating a colour with each team
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


field_made= pd.read_csv('pipeline\\field_made')
field_made_level =field_made[(field_made['Margin at Goal']==0)&(field_made['Time']>40)]
fml_wins = field_made_level.loc[field_made_level['Result']=='Won','Time']
fml_ties = field_made_level.loc[field_made_level['Result']=='Tied','Time']
fml_losses =  field_made_level.loc[field_made_level['Result']=='Lost','Time'] 


fig=plt.figure() #set up the figures
fig.set_size_inches(6, 5)
ax=fig.add_subplot(1,1,1)
plt.hist([fml_wins,fml_ties,fml_losses],stacked=True,bins=np.arange(65.5,81.5))
plt.legend(['Won','Tied','Lost'])
plt.xlabel('Time (mins)')
plt.ylim(0,11)
plt.title('Results of field goals kicked when scores level')

fig.savefig('images\\field_goal_level.png')


fig=plt.figure() #set up the figures
fig.set_size_inches(6, 5)
ax=fig.add_subplot(1,1,1)
plt.hist(games['Tries+'],bins=np.arange(-0.5,13.5),normed=True)
plt.plot(np.arange(0,13),poisson.pmf(np.arange(0,13),np.mean(games['Tries+'])))
plt.legend(['Expected if Poisson','Observed'])
plt.xlabel('Number of tries')
plt.title('Tries: obxerved vs expected')
fig.savefig('images\\poisson_try.png')


f = np.unique(field_made['Margin at Goal'],return_counts=True)

fig=plt.figure() #set up the figures
fig.set_size_inches(6, 5)
ax=fig.add_subplot(1,1,1)
plt.bar(f[0],f[1],color='green')
plt.xlim(-0.7,21)
plt.xlabel('Margin at Goal')
plt.ylabel('No. of Field Goals')
plt.xticks(np.arange(0,24,6))

fig.savefig('images\\field_goal_margins.png')

from sklearn import linear_model
from sklearn import metrics
x = np.array(field_made_level['Time']).reshape(-1,1)
y = np.array(field_made_level['Result']=='Won')
lr = linear_model.LogisticRegression(solver='newton-cg').fit(x, y)
    
lr.intercept_
lr.coef_


#example of scoreline
s=get_scoreline('Sharks vs Raiders',2019)
def plot_score(scoreline):
    fig, ax = plt.subplots()
    home=ax.plot(scoreline['Time'],scoreline.iloc[:,1],color=team_cols[scoreline.columns[1]])
    away=ax.plot(scoreline['Time'],scoreline.iloc[:,2],color=team_cols[scoreline.columns[2]])
    plt.title(match+' ' +str(year))
    plt.xlabel('Match Time (mins)')
    plt.ylabel('Score')
    plt.legend([scoreline.columns[1],scoreline.columns[2]])
    plt.show()
    return fig
fig = plot_score(s)
fig.savefig('images\\Sharks_Raiders_scoreline.png')
plt.clf()

def all_times(event):
    a=match_timelines[event].apply(pd.Series).stack().reset_index(drop=True)
    a=a[a<=80]
    return a

#See the distribution of scoring events
fg=all_times('Drop Goal-Made')
fig=plt.figure() #set up the figures
fig.set_size_inches(6, 5)
ax=fig.add_subplot(1,1,1)
plt.hist(fg,bins=np.arange(0.01,82.01,2))
plt.xlabel('Game Time (mins)')
plt.title('Distributions of Drop Goals Over 80 Minutes')
plt.show()
fig.savefig('images\\field_goal_hist.png')

plt.clf()

pg=all_times('Penalty Shot-Made')
fig=plt.figure() #set up the figures
fig.set_size_inches(6, 5)
ax=fig.add_subplot(1,1,1)
plt.hist(pg,bins=np.arange(0.01,82.01,2),color='red')
plt.xlabel('Game Time (mins)')
plt.title('Distributions of Penalty Goals Over 80 Minutes')
plt.show()
fig.savefig('images\\penalty_goal_hist.png')

plt.clf()

tr=all_times('Try')
fig=plt.figure() #set up the figures
fig.set_size_inches(6, 5)
ax=fig.add_subplot(1,1,1)
plt.hist(tr,bins=np.arange(0.01,82.01,2),color='green')
plt.xlabel('Game Time (mins)')
plt.title('Distributions of Tries Over 80 Minutes')
plt.show()
fig.savefig('images\\penalty_goal_hist.png')

plt.clf()


def get_max(x):
    try:
        if max(x)<=80:
            return max(x)
        else:
            x=pd.Series(x)
            return np.max(x[x<=80])
    except:
        return None
#When are the last points scored
e=match_timelines['Try'].map(lambda x: get_max(x))
f=match_timelines['Penalty Shot-Made'].map(lambda x: get_max(x))
g=match_timelines['Drop Goal-Made'].map(lambda x: get_max(x))
last = pd.concat([e,f,g],axis=1)

lasts = last.dropna(how='all')
last = lasts.max(axis=1)
last_score = lasts.idxmax(axis=1) 
x=[last[last_score=='Try'],last[last_score=='Penalty Shot-Made'],last[last_score=='Drop Goal-Made']]
plt.hist(x,bins=80,stacked=True)
plt.legend(['Try','Penalty','Field Goal'])




try_rate = 80/(tr.shape[0]/games.shape[0])
exp_times = np.exp(-np.linspace(0,81,80)/try_rate)
plt.plot(np.linspace(0,81,80),exp_times*games.shape[0])
plt.hist(80-e[e>0])

def get_min(x):
    try:
        if min(x)<=80:
            return min(x)
        else:
            return 81
    except:
        return 81

team_timelines = match_timelines.loc[match_timelines['Team']!='']

e1=team_timelines['Try'].map(lambda x: get_min(x))
plt.hist(e1,bins=80,cumulative=True,density=True)
plt.plot(np.linspace(0,81,80),(1-exp_times))
plt.xlabel("Time until first try")


pen_rate = 80/(pg.shape[0]/match_timelines[match_timelines['Team']!=''].shape[0])
cum_pen_exp =1- np.exp(-np.linspace(0,81,80)/pen_rate)

f1=team_timelines['Penalty Shot-Made'].map(lambda x: get_min(x))
plt.hist(f1,bins=80,cumulative=True,density=True)
plt.plot(np.linspace(0,81,80),cum_pen_exp)
plt.ylim(0,0.6)
plt.xlabel("Time until first pealty goal")

field_rate = 80/(fg.shape[0]/games.shape[0])
cum_field_exp =1- np.exp(-np.linspace(0,81,80)/field_rate)

g1=team_timelines['Drop Goal-Made'].map(lambda x: get_min(x))
plt.hist(g1,bins=80,cumulative=True,density=True)
plt.plot(np.linspace(0,81,80),cum_field_exp)
plt.ylim(0,0.2)
plt.xlabel("Time until first field goal")

g=match_timelines['Drop Goal-Made'].map(lambda x: get_max(x))


#Plotting probabiliy heat map for successfully kicking a field goal
a=np.arange(1,50)
b=np.arange(-33.5,34.5)
(A,B)=np.meshgrid(a,b)




dist=np.sqrt(A**2+B**2)

ang_left = np.arctan((B+2.75)/A)
ang_right = np.arctan((B-2.75)/A) 

angle =np.abs(ang_left-ang_right)
sig= 1/(1+np.exp((dist-25)/10))

field_prob = np.zeros((68,60))

field_prob[:,11:]= sig

plt.imshow(field_prob,cmap='RdYlGn_r')

plt.imshow(angle*5,cmap='RdYlGn_r')


d = np.arange(6)
(C,D)=np.meshgrid(a,d)

try_prob = 0.8*np.exp(-C/20)*np.power(0.95,D)

plt.imshow(try_prob,cmap='RdYlGn_r')

def draw_pitch(ax,fill=True,numbers=True):
    # focus on only half of the pitch
    #Pitch Outline & Centre Line
    if fill:
        Pitch = pat.Rectangle([0,0], width = 60, height = 68, facecolor='green', edgecolor='white')
    else:
        Pitch = pat.Rectangle([0,0], width = 60, height = 68, fill=False)
    #Left, Right Penalty Area and midline
    goalline =pat.ConnectionPatch([10,0], [10,68], "data", "data",color='white',lw=4)
    tenline =pat.ConnectionPatch([20,0], [20,68], "data", "data",color='white')
    twentyline =pat.ConnectionPatch([30,0], [30,68], "data", "data",color='white')
    thirtyline =pat.ConnectionPatch([40,0], [40,68], "data", "data",color='white')
    fortyline =pat.ConnectionPatch([50,0], [50,68], "data", "data",color='red')
    midline = pat.ConnectionPatch([60,0], [60,68], "data", "data",color='white')
    
    #goalposts
    leftpost=pat.ConnectionPatch([10,30], [2,40],"data","data",color='white')
    rightpost=pat.ConnectionPatch([10,38], [2,48],"data","data",color='white')
    crossbar = pat.ConnectionPatch([7,33.65], [7,41.65],"data","data",color='white')
    thedot = pat.ConnectionPatch([7,37.5], [7,37.8],"data","data",color='black')


    element = [Pitch, goalline,tenline,twentyline,thirtyline,fortyline,midline,leftpost,rightpost,crossbar,thedot]
    for i in element:
        ax.add_patch(i)
    if numbers==True:     
        plt.text(18,10,'1',color='white',fontsize=17)
        plt.text(20,10,'0',color='white',fontsize=17)
        plt.text(27,10,'2',color='white',fontsize=17)
        plt.text(30,10,'0',color='white',fontsize=17)
        plt.text(37,10,'3',color='white',fontsize=17)
        plt.text(40,10,'0',color='white',fontsize=17)
        plt.text(47,10,'4',color='white',fontsize=17)
        plt.text(50,10,'0',color='white',fontsize=17)

        
fig=plt.figure() #set up the figures
fig.set_size_inches(6, 6.8)
ax=fig.add_subplot(1,1,1)
draw_pitch(ax) #overlay our different objects on the pitch
plt.ylim(-2, 70)
plt.xlim(-2, 62)
plt.axis('off')
plt.show()

fig.savefig('images\\pitch.png')


fig=plt.figure() #set up the figures
fig.set_size_inches(7, 5)
ax=fig.add_subplot(1,1,1)
draw_pitch(ax,fill=False) #overlay our different objects on the pitch
heat =plt.imshow(field_prob,cmap='RdYlGn_r')
fig.colorbar(heat)
plt.ylim(0, 68)
plt.xlim(0, 60)
plt.axis('off')
plt.title('Probability of Successfully Kicking a Field Goal')
plt.show()

fig.savefig('images\\field_probs.png')


#

d = np.arange(6)
(C,D)=np.meshgrid(a,d)

try_prob = 0.5*np.exp(-C/20)*np.power(0.9,D)


fig=plt.figure() #set up the figures
fig.set_size_inches(7, 5)
ax=fig.add_subplot(1,1,1)
heat=plt.imshow(try_prob,cmap='RdYlGn_r',aspect='auto')
fig.colorbar(heat)

plt.xlabel('Distance from Tryline (m)')
plt.ylabel('Tackle Count')
plt.title('Probability of Scoring a Try')
plt.show()

fig.savefig('images\\try_probs.png')

plt.clf()



raiders_win = pd.read_csv('pipeline\\raiders_win',header=None)

def decider(win_probs, t, tackle):
    win_prob_if_attempt = field_prob[:,11:]*win_probs.iloc[t-60,0] + (1-field_prob[:,11:])*win_probs.iloc[t-60,2]
    win_prob_if_no_attempt = try_prob[tackle,:]*win_probs.iloc[t-60,1] + (1-try_prob[tackle,:])*win_probs.iloc[t-60,2] 
    
    ar = np.zeros((68,60))
    ar[:,11:] = win_prob_if_attempt-win_prob_if_no_attempt
    fig=plt.figure() #set up the figures
#    fig.set_size_inches(7, 5)
#    ax=fig.add_subplot(1,1,1)
#    draw_pitch(ax,fill=False) #overlay our different objects on the pitch
#    cmap = plt.cm.RdYlGn_r
#    cmap.set_under(color='green')
#    heat =plt.imshow(ar,aspect='auto',cmap=cmap,vmin=0.001,vmax=0.25)
#    plt.ylim(0, 68)
#    plt.xlim(0, 60)
#    plt.axis('off')
    return ar, fig
cmap = plt.cm.RdYlGn_r
cmap.set_under(color='green')
fig,axes=plt.subplots(4,3) #set up the figures
#fig.set_size_inches(7, 5)
for i in np.arange(4):
    for j in np.arange(3):
        ax=axes[i,j]
        draw_pitch(ax,fill=False,numbers=False)
        ax.imshow(decider(raiders_win,62+5*i,1+2*j)[0],aspect='auto',cmap=cmap,vmin=0.001,vmax=0.25,origin='lower')
        ax.axis('off')


cols = 1+2*np.arange(3)
rows = 62+5*np.arange(4)


for ax, col in zip(axes[-1], cols):
    ax.annotate(col, xy=(0.5, 0), xytext=(0, -10),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

for ax, row in zip(axes[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad, 0),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='right', va='center')

fig.text(0.5, 0, 'Tackle Count',size='large', ha='center')
fig.text(0, 0.5, 'Time', va='center',size='large', rotation='vertical')
fig.tight_layout()
plt.show()
fig.savefig('images\\cases.png')

import scipy as scipy
av_tries=np.mean(games['Tries+'])

def exp_tries(game, av_tries,home=True):
    exp_tries = game.loc[game['Home']==home,'tries_scored_per_game'].iloc[0]*game.loc[game['Home']!=home,'tries_conceded_per_game'].iloc[0]/av_tries
    return exp_tries

exp_home_tries = games.groupby(['Year','Match'],as_index=False).apply(lambda x:exp_tries(x,av_tries,home=True))
exp_away_tries = games.groupby(['Year','Match'],as_index=False).apply(lambda x:exp_tries(x,av_tries,home=False))

exp_tries = exp_home_tries.reset_index().merge(exp_away_tries.reset_index(),on=['Year','Match'])

home_games=games.loc[games['Home']]
exp_obs_tries = exp_tries.merge(home_games[['Year','Match','Tries+','Tries-']],on=['Year','Match'])
mat=exp_obs_tries[exp_obs_tries['Year']>2013]

mat.to_csv('pipeline\\exp_obs')
chi_stat =   (exp_obs_tries.iloc[:,4]-exp_obs_tries.iloc[:,2])**2/exp_obs_tries.iloc[:,2]

obs_home = mat.iloc[:,4]
exp_home = mat.iloc[:,2]

home_dev = 2*(obs_home*np.log(obs_home/exp_home)-obs_home+exp_home)
home_dev[obs_home==0]=2*exp_home

obs_away = mat.iloc[:,5]
exp_away = mat.iloc[:,3]

away_dev = 2*(obs_away*np.log(obs_away/exp_away)-obs_away+exp_away)
away_dev[obs_away==0]=2*exp_away



pennies= games[['PenaltyGoals+','penalty_goals_per_game']].iloc[384:]
pennies.to_csv('pipeline\\exp_obs_pen')

import scipy.stats as stats

1-scipy.stats.chi2.cdf(np.sum(home_dev), 1151)

1-scipy.stats.chi2.cdf(np.sum(away_dev), 1151)

1-scipy.stats.chi2.cdf(np.sum(away_dev)+np.sum(home_dev), 2304)

1-scipy.stats.chi2.cdf(2348.2, 2302)


pens =  match_timelines.loc[match_timelines['Team']!='','Penalty Shot-Made']

times=[]
events=[]
for i in np.arange(pens.shape[0]):
    if np.isnan(pens[i]).any():
        times.append(80.0)
        events.append(0)
    else:
        x=pens[i]
        x.sort()
        for j in np.arange(len(x)):
            if j==0:
                times.append(x[j])
                events.append(1)
            else:
                times.append(x[j]-x[j-1])
                events.append(1)
                
        times.append(80-x[-1])
        events.append(0)
        
p = pd.DataFrame({'time':times,'Goal':events})
p1=p.loc[(p['time']>0)&(p['time']<=80)] 

p1.to_csv('pipeline\\penalty')
p1.sort_values(by='time',inplace=True)
p1['Risk']= p1.shape[0]-np.arange(p1.shape[0],)
p1['Cond_no_goal']=(p1['Risk']-p1['Goal'])/p1['Risk']
p1['SurvProb']=np.cumprod(p1['Cond_no_goal'])

plt.plot(p1['time'],1-p1['SurvProb'])
plt.plot(p1['time'],1-np.exp(-p1['time']/pen_rate))
plt.legend(['Observed','Expected'])

trs =  match_timelines.loc[match_timelines['Team']!='','Try']

times=[]
events=[]
for i in np.arange(trs.shape[0]):
    if np.isnan(trs[i]).any():
        times.append(80.0)
        events.append(0)
    else:
        x=trs[i]
        x.sort()
        for j in np.arange(len(x)):
            if j==0:
                times.append(x[j])
                events.append(1)
            else:
                times.append(x[j]-x[j-1])
                events.append(1)
                
        times.append(80-x[-1])
        events.append(0)


t = pd.DataFrame({'time':times,'Goal':events})
t1=t.loc[(t['time']>0)&(t['time']<=80)] 

t1.to_csv('pipeline\\try')
t1.sort_values(by='time',inplace=True)
t1['Risk']= t1.shape[0]-np.arange(t1.shape[0],)
t1['Cond_no_goal']=(t1['Risk']-t1['Goal'])/t1['Risk']
t1['SurvProb']=np.cumprod(t1['Cond_no_goal'])

try_rate = 80/(tr.shape[0]/match_timelines[match_timelines['Team']!=''].shape[0])

fig=plt.figure() #set up the figures
fig.set_size_inches(6,5)
ax=fig.add_subplot(1,1,1)
plt.plot(t1['time'],t1['SurvProb'])
plt.plot(t1['time'],np.exp(-t1['time']/try_rate))
plt.legend(['Observed','Expected'])
plt.title('Survival until next try: observed vs expected')
fig.savefig('images\\try_srv.png')

ha =np.array(home_games[['Tries+','Tries-']])
hh=[]
for i in np.arange(np.max(ha,axis=0)[0]+1):
    hh.append(np.unique(ha[ha[:,0]==i,1],return_counts=True))

hj = np.zeros((np.max(ha,axis=0)+1))
for i in np.arange(np.max(ha,axis=0)[0]+1):
    for j in np.arange(len(hh[i][0])):
        hj[i,hh[i][0][j]] = hh[i][1][j]

fact_h = np.ones(hj.shape[0])        
fact_h[1:]=np.cumprod(np.arange(1,hj.shape[0]))
fact_a = np.ones(hj.shape[1])        
fact_a[1:]=np.cumprod(np.arange(1,hj.shape[1]))

mean_home= np.mean(ha[:,0])
mean_away = np.mean(ha[:,1])
ex_h = mean_home**(np.arange(hj.shape[0]))*np.exp(-mean_home)/fact_h
ex_a = mean_away**(np.arange(hj.shape[1]))*np.exp(-mean_away)/fact_a

exp_ha =np.outer(ex_h,ex_a)*ha.shape[0]
cf = np.corrcoef(ha.T)


#Using Skellam distribution to find probabilty of victory if field goal is kicked
seconds = np.linspace(0,80,4801)
new_mean = (80-seconds)/try_rate

from scipy.stats import skellam


def prob_win(n):
    z=np.zeros((n+1,4801))
    for i in np.arange(n+1):
        z[i,:]=skellam.cdf(i,new_mean,new_mean)
    return z

z= prob_win(3)


def prob_win_change(n):
    z=np.zeros((n+1,4801))
    for i in np.arange(n+1):
        z[i,:]=skellam.pmf(i,new_mean,new_mean)*0.5
    return z

change= prob_win_change(3)
zoomz = z.T[3600:,:]
zoomchange = change.T[3600:,:]
yoomy = zoomz-zoomchange


fig=plt.figure() #set up the figures
fig.set_size_inches(10,10)
ax1=fig.add_subplot(2,2,1)
out_mat = [[0,1,1,1],[-1,0,1,1],[-1,-1,0,1],[-1,-1,-1,0]]
plt.imshow(out_mat, interpolation='nearest',cmap='RdBu')
plt.title('Outcomes')
plt.ylabel('Opposition Tries')
plt.xlabel('Tries')
plt.xticks([0,1,2,3])
plt.yticks([0,1,2,3])
s = [['DW','WW','WW','WW'], ['LL', 'DW','WW','WW'],['LL','LL','DW','WW'],['LL','LL','LL','DW']]
for i in range(4):
    for j in range(4):
        plt.text(j,i, str(s[i][j]))

ax2=fig.add_subplot(2,2,2)
plt.plot(seconds[3600:],yoomy)
plt.title('Probability of victory before field goal')
plt.xlabel('Match time')
plt.legend(['0','6','12','18'],title="Lead by:",loc='lower left')
plt.ylim(0.48,1.02)
ax23=fig.add_subplot(2,2,3)
plt.plot(seconds[3600:],zoomz)
plt.title('Probability of victory after field goal')
plt.xlabel('Match time')
plt.ylim(0.48,1.02)
ax4=fig.add_subplot(2,2,4)
plt.plot(seconds[3600:],zoomchange)
plt.title('Increase in probability of victory after field goal')
plt.xlabel('Match time')
fig.savefig('images\\vic_prob.png')



plt.clf()



fig=plt.figure() #set up the figures
fig.set_size_inches(6, 6.8)
ax=fig.add_subplot(1,1,1)


fig=plt.figure() #set up the figures
ax=fig.add_subplot(1,1,1)
plt.plot(np.arange(60,80),raiders_win)
plt.legend(['Win field goal','Win try','Win no score'])
plt.title('Probability of Raiders Victory')
plt.xlabel('Time')

fig.savefig('images\\vic_probs.png')
