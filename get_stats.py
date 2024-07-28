# -*- coding: utf-8 -*-
"""

"""

from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC 
from selenium.common.exceptions import TimeoutException
import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver.common.action_chains import ActionChains
import os 

#chrome_options = Options()
# 
#chrome_options.add_argument("--headless")
#chrome_options.add_argument('--disable-gpu')
#chrome_options.add_argument('--log-level=3')




chrome_options = webdriver.ChromeOptions()

chrome_options.page_load_strategy = 'normal'
    
chrome_options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=chrome_options)




def get_player_stats(table):

    n_columns = 0
    n_rows=0
    column_names = []

    for row in table.find_all('tr')[2:]:
        # Determine the number of rows in the table
        td_tags = row.find_all('td')
        if len(td_tags) > 0:
            n_rows+=1
            if n_columns == 0:
            # Set the number of columns for our table
                n_columns = len(td_tags)

        # Handle column names ifget_player_stats we find them
        th_tags = row.find_all('th') 
        if len(th_tags) > 0 and len(column_names) == 0:
            for th in th_tags:
                column_names.append(th.get_text())

    # Safeguard on Column Titles
    if len(column_names) > 0 and len(column_names) != n_columns:
        raise Exception("Column titles do not match the number of columns")

    columns = column_names if len(column_names) > 0 else range(0,n_columns)
    df = pd.DataFrame(columns = columns,
                                  index= range(0,n_rows))
    row_marker = 0
    for row in table.find_all('tr'):
        column_marker = 0
        columns = row.find_all('td')
        for column in columns:
            df.iat[row_marker,column_marker] = column.get_text().strip()
            column_marker += 1
        if len(columns) > 0:
            row_marker += 1
    #Get column names
    header=table.find_all('tr')[1]
    for th in header.find_all('th'):
        column_names.append(th.get_text())
    df.columns=column_names
    
    #Get team name
    df['Team']=table.find('caption').get_text().strip().split(' ', 1)[0]
    

    
    
    # Convert to float if possible
    for col in df:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass
    return df




def get_all_player_stats(soup):
      
    df=pd.DataFrame()
    for table in soup.find_all('table'):
        newdf=pd.DataFrame(get_player_stats(table))
        df=pd.concat([df,newdf])
    df=df.drop_duplicates()
    #Get Round Number and date
    df['Round']=soup.find("p", class_="match-header__title").get_text().split(' ')[1]
    df['date']=soup.find("time")['datetime']
    return df




def get_timeline(soup):
    events=soup.find_all("div", class_="match-centre-event__content")
    data=[]
    for event in events:
        time= event.find(class_="match-centre-event__timestamp").get_text().strip() if event.find(class_="match-centre-event__timestamp") else None
        event_type = event.find(class_="match-centre-event__title").get_text().strip() if event.find("h4", class_="match-centre-event__title") else None
        player = event.find('p',class_="u-font-weight-500").get_text().strip() if event.find('p',class_="u-font-weight-500") else None
        team = event.find(class_="match-centre-event__team-name").get_text().strip() if event.find(class_="match-centre-event__team-name") else None
        player_on = event.find(class_="match-centre-event__interchange-player match-centre-event__interchange-player--on").parent.get_text().strip() if event.find(class_="match-centre-event__interchange-player match-centre-event__interchange-player--on") else None
        player_on = player_on.split(' ',1)[1] if player_on else None
        player_off= event.find(class_="match-centre-event__interchange-player match-centre-event__interchange-player--off").parent.get_text().strip() if event.find(class_="match-centre-event__interchange-player match-centre-event__interchange-player--off") else None
        player_off = player_off.split(' ',1)[1] if player_off else None
        score= event.find(class_="match-centre-event__score").get_text().strip() if event.find(class_="match-centre-event__score") else None
        score = score.replace("\n","").replace(' ','') if score else None
        dic = {"Time":time,"Event Type":event_type,"Player":player, "Team":team, "Player On":player_on,"Player Off":player_off,"Score":score}
        data.append(dic)
        
    df1=pd.DataFrame(data)
    df1.loc[df1['Event Type'].str.contains("Interchange"),'Team']=df1['Event Type'].str.split(' ').str.get(0)
    df1.loc[df1['Event Type'].str.contains("Interchange"),'Event Type']=df1['Event Type'].str.split(' ',n=1).str.get(1)
    
    return df1


def get_team_stats(soup):
    j ={}
    stats = soup.find_all('div', class_="u-spacing-pb-24 u-spacing-pt-16 u-width-100")
    for stat in stats:
        text=stat.find(class_="stats-bar-chart__title").get_text().strip() if stat.find( class_="stats-bar-chart__title") else None
        
        k=[]
        te = stat.find_all('p')
        for t in te:
            k.append(t.get_text().strip())
    
        de=stat.find_all('dd')
        for d in de:
            k.append(d.get_text().strip())
        l =list(filter(lambda a: a != '', k))
        j[text]=l
    
    try:
        j['Completions (%)']= list(filter(lambda a: '%' in a, j['Completion Rate']))    
    except:
        j['Completion (%)']='na'
    try:
        j['Completed Sets Ratio']=list(filter(lambda a: '/' in a, j['Completion Rate']))   
    except:
        j['Completed Sets Ratio']='na'
    try:
        j.pop('Completion Rate', None)
    except:
        pass
    for k in j.keys():
        j[k]=j[k][0:2]
        
        
    info = pd.DataFrame(j)   
    return info






def get_game_stats(url):
        
    options = webdriver.ChromeOptions()

    options.page_load_strategy = 'normal'
    
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver,30)
    #Go to game website
    driver.get(url)
    driver.implicitly_wait(30)
    #Wait until 'Latest' tab then click
    chart = wait.until(EC.visibility_of_element_located((By.XPATH,'//*[@id="tabs-match-centre-"]/div[1]/div/div/ul/li[2]')))
    driver.execute_script("arguments[0].scrollIntoView(true)",chart)

    element = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="tabs-match-centre-"]/div[1]/div/div/ul/li[2]')))
    ActionChains(driver).move_to_element(element).click().perform()
    #element.click()
    soup = BeautifulSoup(driver.page_source, 'lxml')
    #Get names of home and away teams
    home = soup.find('p', class_="match-team__name match-team__name--home").get_text().strip()
    away = soup.find('p', class_="match-team__name match-team__name--away").get_text().strip()
    match_name = home+' vs '+away
    
    #Get timeline stats
    df1 = get_timeline(soup)
    df1['Match'] = match_name
    driver.get(url)
    driver.implicitly_wait(30)
    chart = wait.until(EC.visibility_of_element_located((By.XPATH,'//*[@id="tabs-match-centre-"]/div[1]/div/div/ul/li[5]')))
    driver.execute_script("arguments[0].scrollIntoView(true)",chart)
    #Go to 'player stats' tab
    driver.implicitly_wait(30)
    element = WebDriverWait(driver, 20).until(
    EC.element_to_be_clickable((By.XPATH, '//*[@id="tabs-match-centre-"]/div[1]/div/div/ul/li[5]')))
    ActionChains(driver).move_to_element(element).click().perform()

    
    #element.click()
    soup = BeautifulSoup(driver.page_source, 'lxml')
    
    #get individual player stats
    df2=get_all_player_stats(soup)
    df2['Match'] = match_name
    
    
    #Go to 'team stats' tab
    chart = wait.until(EC.visibility_of_element_located((By.XPATH,'//*[@id="tabs-match-centre-"]/div[1]/div/div/ul/li[4]')))
    driver.execute_script("arguments[0].scrollIntoView(true)",chart)
    element = WebDriverWait(driver, 20).until(
    EC.element_to_be_clickable((By.XPATH, '//*[@id="tabs-match-centre-"]/div[1]/div/div/ul/li[4]')))
    ActionChains(driver).move_to_element(element).click().perform()

    #element.click()
    soup = BeautifulSoup(driver.page_source, 'lxml')
    
    #get team stats
    df3 = get_team_stats(soup)
    df3['Match']= match_name
    df3['Team']=pd.Series([home,away])
    
    driver.quit()
    return match_name,df1,df2,df3



def get_round(url):
    match_urls=[]
    games=[]
    driver = webdriver.Chrome(options=chrome_options)
    
    #Go to game website
    driver.get(url)
    driver.implicitly_wait(30)
    soup = BeautifulSoup(driver.page_source, 'lxml')
    for game in soup.find_all("a", class_="match--highlighted u-flex-column u-flex-align-items-center u-width-100"):
        match_urls.append(game['href'])

    for match_url in match_urls:
        print(match_url)
        match_name,df1,df2,df3 = get_game_stats('https://www.nrl.com'+match_url)
        g = {'Match':match_name,'Timeline':df1,'Player Stats':df2,'Team':df3}
        games.append(g)
    
    return games
    

def get_year(year, start=0, stop=31):
    path = os.getcwd() + '\\data'
    
    yearpath= path + '\\' + str(year)
    if not os.path.exists(yearpath):
        os.makedirs(yearpath)
    #try:  
    #    os.mkdir(yearpath)
    #except OSError:  
    #    print ("Creation of the directory %s failed" % yearpath)
    #except print ("Directory %s already exists" % yearpath))
    #else:  
    #    print ("Successfully created the directory %s" % yearpath)
        
    for i in range(start,stop):
        url= 'https://www.nrl.com/draw/?competition=111&round='+str(i+1)+'&season='+str(year)
        
        roundpath = yearpath +'\\' +  'Round'+str(i+1)
        
        try:  
            os.mkdir(roundpath)
        except OSError:  
            print ("Creation of the directory %s failed" % roundpath)
            continue
        else:  
            print ("Successfully created the directory %s" % roundpath)
        try: 
            round_stats = get_round(url)   
        except:
            pass
        for item in round_stats:
            matchpath = roundpath +'\\'+ item['Match'].replace(' ','')
            item['Player Stats']['Year']=year
            item['Timeline']['Year']=year
            item['Team']['Year']=year
            try:  
                os.mkdir(matchpath)
            except OSError:  
                print ("Creation of the directory %s failed" % matchpath)
                continue
              
            print ("Successfully created the directory %s" % matchpath)
                
            player_path = matchpath + '\\player.csv'
            item['Player Stats'].to_csv(player_path)
            timeline_path = matchpath + '\\timeline.csv'
            item['Timeline'].to_csv(timeline_path)
            team_path = matchpath + '\\team.csv'
            item['Team'].to_csv(team_path)

#Get stats from 2013 to 2023

try:
    os.mkdir(os.getcwd() + '\\data')
except:
    pass

for i in range(2013,2024):
    get_year(i)
    
