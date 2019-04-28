from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re

# scrapes the teams in the east and the west (by abbreviation)
YEAR = 2019
url = 'https://www.basketball-reference.com/leagues/NBA_{}_standings.html'.format(YEAR)
html = urlopen(url)
soup = BeautifulSoup(html, 'lxml')
print("Generating team abbreviations for East")
head = soup.find(id='div_confs_standings_E')
names = [link.get('href')[7:10] for link in head.findAll('a', attrs={'href': re.compile("^/teams/")})]
names = sorted(names)
names = pd.DataFrame(names, columns=['teams'])
names.to_csv("./teamScrapedData/eastTeams.csv")
print("Generating team abbreviations for West")
head = soup.find(id='div_confs_standings_W')
names = [link.get('href')[7:10] for link in head.findAll('a', attrs={'href': re.compile("^/teams/")})]
names = sorted(names)
names = pd.DataFrame(names, columns=['teams'])
names.to_csv("./teamScrapedData/westTeams.csv")