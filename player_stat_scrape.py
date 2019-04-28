from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re

YEAR = 2019
# map full team name to 3 letter name to index
teamvteam_data = pd.read_csv("./teamScrapedData/teamvteam_{}.csv".format(YEAR))
abbr_names = teamvteam_data.columns[2:]
abbr_names = abbr_names.values.tolist()

# go through all the teams and pull players
for i in range(len(abbr_names)):
	team_abbr = abbr_names[i]
	print("Extracting Player +/- for {}--------".format(team_abbr))
	url = 'https://www.basketball-reference.com/teams/{}/{}.html'.format(team_abbr, YEAR)
	html = urlopen(url)
	soup = BeautifulSoup(html, "lxml")
	
	head = soup.find(id='div_roster')
	num_players = len(head.findAll('a', attrs={'href': re.compile("^/players/")}))
	team_data = pd.DataFrame(columns=['Name']+abbr_names)
	team_data_games = pd.DataFrame(columns=['Name']+abbr_names)
	team_data_mins = pd.DataFrame(columns=['Name']+abbr_names)
	plusminus_players = np.zeros((num_players,))

	# iterate through the players
	for link in head.findAll('a', attrs={'href': re.compile("^/players/")}):
		player_link = link.get('href')[:-5]
		player_name = link.getText()
		player_stats = {'Name': player_name}
		player_stats2 = {'Name': player_name}
		player_stats3 = {'Name': player_name}
		# initialize the dictionary
		for i in range(len(abbr_names)):
			player_stats[abbr_names[i]] = 0
			player_stats2[abbr_names[i]] = 0
			player_stats3[abbr_names[i]] = 0
		print("{} {}".format(player_name, player_link))
		player_url = 'https://www.basketball-reference.com{}/splits/{}'.format(player_link, YEAR)
		html2 = urlopen(player_url)
		soup2 = BeautifulSoup(html2, 'lxml')

		# go through all opposing teams
		head_pm = soup2.find(id='div_splits')
		if head_pm != None:
			for entry in head_pm.findAll('a', attrs={'href': re.compile("^/teams/")}):
				opp_team = entry.get('href')[7:10]
				row = entry.parent.parent
				pm = float(row.find('td', attrs={'data-stat': 'plus_minus_per_200_poss'}).getText())
				num = float(row.find('td', attrs={'data-stat': 'g'}).getText())
				mins = float(row.find('td', attrs={'data-stat': 'mp'}).getText())
				player_stats[opp_team] = pm
				player_stats2[opp_team] = num
				player_stats3[opp_team] = mins
		team_data = team_data.append(player_stats, ignore_index=True)
		team_data_games = team_data_games.append(player_stats2, ignore_index=True)
		team_data_mins = team_data_mins.append(player_stats3, ignore_index=True)
	# export to CSV
	team_data.to_csv('./playerScrapedData/{}_pm{}.csv'.format(team_abbr, YEAR))
	team_data_games.to_csv('./playerScrapedData/{}_g{}.csv'.format(team_abbr, YEAR))
	team_data_mins.to_csv('./playerScrapedData/{}_mp{}.csv'.format(team_abbr, YEAR))
