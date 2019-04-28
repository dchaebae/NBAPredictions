from urllib.request import urlopen
from bs4 import BeautifulSoup, Comment
import pandas as pd
import numpy as np

def scraper(query, year=2019):
	""" Scrape NBA data from basketball-reference.com
	"""
	for i in range(query.shape[0]):
		url = query[i,0].format(year)
		html = urlopen(url)
		soup = BeautifulSoup(html, "lxml")
		print("Generating {} using {}".format(query[i,1].format(year), query[i,2]))

		# Grab the headers
		html = None
		head = soup.find(id=query[i,2])
		head_ind = int(query[i,3]) # start of header index
		# Case 1: Check to see if delineated by comments
		for c in head.children:
			if isinstance(c, Comment):
				html = c
		if html is not None:
			soup = BeautifulSoup(html, "lxml")
			headers = [th.getText() for th in soup.find_all('tr', limit=head_ind+1)[head_ind].find_all('th')]
		# Case 2: Not delineated by comments
		else:
			headers = [th.getText() for th in head.find_all('tr', limit=head_ind+1)[head_ind].find_all('th')]
		headers = headers[1:]

		# Grab the data
		rows = soup.findAll('tr')[head_ind+1:]
		stats = [[td.getText() for td in rows[i].findAll('td')]
		            for i in range(len(rows))]

		stats = pd.DataFrame(stats, columns = headers)
		if 'april' not in query[i,1]:
			stats.to_csv("./teamScrapedData/{}".format(query[i,1].format(year)))
		else:
			stats.to_csv("./teamScrapedData/test{}".format(query[i,1].format(year)))

query = np.genfromtxt('scrape.txt', dtype='str')
scraper(query)