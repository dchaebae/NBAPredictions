import numpy as np
import pandas as pd
import glob

# Iterate through PageRank algorithm using G matrix, matrix, and 
# the initial importance vector
def converger(importance, matrix):
	while True:
		next_importance = np.matmul(importance, matrix)
		l1_dist = np.sum(abs(importance-next_importance))
		importance = next_importance
		if l1_dist < 1e-10:
			break
	return importance

YEAR = 2019
teamvteam_data = pd.read_csv("./teamScrapedData/teamvteam_{}.csv".format(YEAR))
abbr_names = teamvteam_data.columns[2:]
full_names = teamvteam_data['Team']
num_teams = abbr_names.shape[0]
abbr_ind = dict(zip(abbr_names, range(num_teams)))

team_files = sorted(glob.glob("./playerScrapedData/*_pm{}.csv".format(YEAR)))
team_list = []
# combine all the months to one Dataframe
for filename in team_files:
	df = pd.read_csv(filename)
	team_list.append(df)

team_files = sorted(glob.glob("./playerScrapedData/*_g{}.csv".format(YEAR)))
game_list = []
# combine all the months to one Dataframe
for filename in team_files:
	df = pd.read_csv(filename)
	game_list.append(df)

team_files = sorted(glob.glob("./playerScrapedData/*_mp{}.csv".format(YEAR)))
min_list = []
# combine all the months to one Dataframe
for filename in team_files:
	df = pd.read_csv(filename)
	min_list.append(df)

print('===================================================')
print('Pairwise Comparisons')
print('===================================================')
query = np.genfromtxt('matchups.txt', dtype='str')
for k in range(query.shape[0]):
	(team_A, team_B) = query[k]
	print('-----{} vs {}-----'.format(team_A, team_B))
	data_A = team_list[abbr_ind[team_A]]
	games_A = game_list[abbr_ind[team_A]][team_B]
	time_A = min_list[abbr_ind[team_A]][team_B]

	data_B = team_list[abbr_ind[team_B]]
	games_B = game_list[abbr_ind[team_B]][team_A]
	time_B = min_list[abbr_ind[team_B]][team_A]

	names = data_A['Name'].append(data_B['Name'])
	names.index = range(names.shape[0])
	names_list = names.to_numpy()
	pm_A = data_A[team_B]
	pm_B = data_B[team_A]

	num_A = data_A.shape[0]
	num_B = data_B.shape[0]
	mat_size = num_A+num_B
	rank_matrix = np.zeros((mat_size, mat_size))
	for i in range(0,num_A):
		for j in range(num_A,mat_size):
			num_i = games_A[i]
			num_j = games_B[j-num_A]
			prop_i = time_A[i] / (num_i * 48) if num_i > 0 else 0
			prop_j = time_B[j-num_A] / (num_j * 48) if num_j > 0 else 0
			temp_i = -10
			temp_j = -10
			if prop_i > 0.354:
				if num_i == 0:
					temp_i = -10
				else:
					scale_A = prop_i ** 2
					temp_i = pm_A[i] * scale_A if pm_A[i] > 0 else pm_A[i] * (2 - scale_A)
			if prop_j > 0.354:
				if num_j == 0:
					temp_j = -10
				else:
					scale_B = prop_j ** 2
					thisB = pm_B[j-num_A]
					temp_j = thisB * scale_B if thisB > 0 else thisB * (2 - scale_B)

			diff = temp_i - temp_j
			if diff > 0:
				rank_matrix[j, i] = diff
			else:
				rank_matrix[i, j] = -diff
	normalizer = np.sum(rank_matrix, axis=1)
	normalizer[normalizer == 0] = 1
	H = rank_matrix /normalizer[:,None]
	importance = np.ones(mat_size)/mat_size
	zero_indices = np.where(~H.any(axis=1))[0] # check for any dangling nodes
	H[zero_indices,:] = importance # make H hat
	t = 0.85
	G = t * H + (1-t)/mat_size*np.ones((mat_size, mat_size))
	importance = converger(importance, G)
	importance_rank = np.argsort(importance)[::-1]
	top5_A = importance_rank[importance_rank < num_A][0:5]
	top5_B = importance_rank[importance_rank >= num_A][0:5]
	print('Top 5 players of {}: {}'.format(team_A, names_list[top5_A]))
	print('Top 5 players of {}: {}'.format(team_B, names_list[top5_B]))
	print('Top 5 players of matchup: {}'.format(names_list[importance_rank[0:5]]))
print('===================================================')
print('All-Star: East vs West')
print('===================================================')
east_teams = pd.read_csv('./teamScrapedData/eastTeams.csv')
num_east = east_teams.shape[0]
west_teams = pd.read_csv('./teamScrapedData/westTeams.csv')
set_east = set(east_teams['teams'])
set_west = set(west_teams['teams'])

team_sizes = np.zeros(num_teams)
team_sizes = team_sizes.astype(int)
team_upper = np.zeros(num_teams)
team_upper = team_upper.astype(int)
ind_team = {}
player_names = team_list[0]['Name']
# figure out number of players on each team
# Also map index to players, index to team
for i in range(num_teams):
	alpha_abbr = abbr_names[i]
	alpha_team = team_list[abbr_ind[alpha_abbr]]
	# number of players
	if alpha_abbr in set_east:
		team_sizes[i] = alpha_team.shape[0]
	elif alpha_abbr in set_west:
		team_sizes[i] = alpha_team.shape[0]

	# cumulative sum for upper index of a team
	if i == 0:
		team_upper[i] = team_sizes[i]
		for k in range(team_sizes[i]):
			ind_team[k] = alpha_abbr
	else:
		team_upper[i] = team_sizes[i] + team_upper[i-1]
		for k in range(team_upper[i-1],team_upper[i]):
			ind_team[k] = alpha_abbr

	# create index to player name dictionary
	if i != 0:
		player_names = player_names.append(alpha_team['Name'])
num_players = team_upper[-1]
player_names.index = range(num_players)
player_names = player_names.to_numpy()

player_matrix = np.zeros((num_players, num_players))
for i in range(num_players):
	team_i = ind_team[i]
	team_ind_i = abbr_ind[team_i]
	clamp = 0 if team_ind_i == 0 else team_upper[team_ind_i-1]

	player_i = team_list[team_ind_i].iloc[i-clamp]
	games_i = game_list[team_ind_i].iloc[i-clamp]
	time_i = min_list[team_ind_i].iloc[i-clamp]

	for j in range(team_upper[team_ind_i], num_players):
		team_j = ind_team[j]
		team_ind_j = abbr_ind[team_j]
		clamp2 = team_upper[team_ind_j-1]

		player_j = team_list[team_ind_j].iloc[j-clamp2]
		games_j = game_list[team_ind_j].iloc[j-clamp2]
		time_j = min_list[team_ind_j].iloc[j-clamp2]

		num_i = games_i[team_j]
		num_j = games_j[team_i]
		ti = time_i[team_j]
		tj = time_j[team_i]

		prop_i = ti / (num_i * 48) if num_i > 0 else 0
		prop_j = tj / (num_j * 48) if num_j > 0 else 0

		temp_i = -10
		temp_j = -10
		if prop_i > 0.354:
			temp_i = -10 if num_i == 0 else player_i[team_j] * prop_i**2
		if prop_j > 0.354:
			temp_j = -10 if num_j == 0 else player_j[team_i] * prop_j**2

		val_diff = temp_i-temp_j
		if val_diff > 0:
			player_matrix[j, i] = val_diff
		else:
			player_matrix[i, j] = -val_diff

normalizer = np.sum(player_matrix, axis=1)
normalizer[normalizer == 0] = 1
H = player_matrix /normalizer[:,None]
importance = np.ones(num_players)/num_players
zero_indices = np.where(~H.any(axis=1))[0] # check for any dangling nodes
H[zero_indices,:] = importance # make H hat
t = 0.85
G = t * H + (1-t)/num_players*np.ones((num_players, num_players))
importance = converger(importance, G)
importance_rank = np.argsort(importance)[::-1]
# determine the best 3 players of each team
# determine the top 10 of east and west conferences
mvp_team = {}
mvp_east = []
mvp_west = []
for i in range(num_players):
	p_team = ind_team[importance_rank[i]]
	p_name = player_names[importance_rank[i]]
	if p_team in set_east:
		if len(mvp_east) < 10:
			mvp_east.append(p_name)
	elif p_team in set_west:
		if len(mvp_west) < 10:
			mvp_west.append(p_name)

	if p_team not in mvp_team or len(mvp_team[p_team]) < 3:
		if p_team not in mvp_team:
			mvp_team[p_team] = []
		mvp_team[p_team].append(p_name)

print('Top 10 players of series: {}'.format(player_names[importance_rank[0:10]]))
print('Best 3 Players of each team:')
print(mvp_team)
print('Best 10 Players of East Conference: {}'.format(mvp_east))
print('Best 10 Players of West Conference: {}'.format(mvp_west))





