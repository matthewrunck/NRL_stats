# NRL_stats
Using rugby league stats from NRL.com to determine the impact of a field goal attempt on the probability of victory, 
conditioned on the circumstances of the game at the time of the attempt.




## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept in the data folder. this folder contains subdirectories for each season, which in turn hold subdirectories for each match. 
Each match folder contains three files 
<pre/>
        Data
        |
        +____Year
                |
                +___Match
                     |
                      +__timeline (a timeline of important events from the match with the game-time at which they occurred)
                      +__player  (individual statistics for each of the 17 players on each team)
                      +__team     (aggregate statistics for each team)
</pre>

This data was obtained from the NRL.com website using selenium webdriver and beautfiul soup to parse the html code and extract relevant data.
This can be replicated by running the 'getdata.py' script if desired.

3. The nnext step is to transform the raw data in order to perform our analysis. In order to do so run these scripts in their numbered order:
    '01_join_games.py'
    This script joins the player team and timeline files for each match into three large dataframes with data for all reglar season matches from 2013-2019

    '02_join_team_players.py'
    aggregates and merges the player and team datasets to produce detailed statisitcs for each game, combining for and against stats (eg points scored vs points conceded)
    into the same observation 
    
    '03_wide_timelines.py'
    creates a 'wide' timeline, with each row corresponding to the events occurring to a team in the match, with each column corresponding to  particular
    event. The values are lists of times that the event occurs to that team in the match.
    
    '04_field_goals.py'
    takes the match timelines and for each match creates a record for the score for each team at each second of the match
    These scorelines are then used to create a dataframe survival analysis, where every score change is treated as an event.
    For each score change the time of the current score, the time of the current score, the margin at the time of the score, whether the score
    change was the result of a field goal is recorded.
    
    '05_field_goals.py'
    information for every field goal attempt, inclding the match-time, scores when field goal was attempted, and the score after 80 minutes 
    
    'decider.py'
    Simulates 
    
    4. required packages:
      selenium
      bs4
      numpy
      pandas
      seaborn
      matplotlib
      





