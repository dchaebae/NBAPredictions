import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

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

# Set to True if you want importance scores printed
PRINT_IMPORTANCE = False
# Set to True if April used as testing set
TEST_SET = False
# Set to True if accuracy is needed in proportion
PROPORTION_ACC = False

# Analysis for 2019 year, can be used for other years as needed
YEAR=2019
games_month = glob.glob("./teamScrapedData/month*_{}.csv".format(YEAR))
data_month = []
# combine all the months to one Dataframe
for filename in games_month:
	df = pd.read_csv(filename)
	data_month.append(df)

# add april games
april_games = pd.read_csv('./teamScrapedData/testmonth_april_2019.csv')
PLAYOFF_SIZE = 79
april_games = april_games.iloc[0:PLAYOFF_SIZE]
data_month.append(april_games)


all_games = pd.concat(data_month, ignore_index=True)
all_games_num = all_games.shape[0]
seq_acc = np.zeros(4+4*2)
min_start = 400

teamvteam_data = pd.read_csv("./teamScrapedData/teamvteam_{}.csv".format(YEAR))
abbr_names = teamvteam_data.columns[2:]
full_names = teamvteam_data['Team']
num_teams = abbr_names.shape[0]
name_dict = dict(zip(full_names, abbr_names))
ind_dict = dict(zip(abbr_names, range(num_teams)))
teamvteam_np = teamvteam_data.to_numpy()[:,2:]

all_acc = np.zeros((all_games_num-min_start,seq_acc.shape[0]))

for test_num in range(min_start,all_games_num):
	print(test_num)
	df = all_games.iloc[0:test_num]
	(visit_name, visit_pointcol, home_name, home_pointcol) = df.columns[2:6]
	# Remove games that have not yet been played (but are included in scrape)
	not_played_ind = np.argwhere(np.isnan(df[home_pointcol]))
	not_played_ind = np.reshape(not_played_ind, not_played_ind.shape[0])
	df = df.drop(not_played_ind)
	df.index = range(df.shape[0])
	
	# ranks for all the different methods we use
	all_rankings = np.zeros((4+4*2, num_teams))

	# make an empty 2d NumPy array to store sums
	# rowPoint-colPoint scoring system
	relation_matrix = np.zeros((num_teams, num_teams))
	visit_teams = df[visit_name]
	visit_points = df[visit_pointcol]
	home_teams = df[home_name]
	home_points = df[home_pointcol]
	diff_points = visit_points-home_points
	for i in range(df.shape[0]):
		visit_ind = ind_dict[name_dict[visit_teams[i]]]
		home_ind = ind_dict[name_dict[home_teams[i]]]
		if diff_points[i] > 0:
			relation_matrix[visit_ind, home_ind] += diff_points[i]
		else:
			relation_matrix[home_ind, visit_ind] += abs(diff_points[i])

	# Run the PageRank Algorithm but for ranking teams, use SUM!
	# A replication of Xia & Jain's weighted method
	normalizer = np.sum(relation_matrix.T, axis=1)
	normalizer[normalizer == 0] = 1
	H = relation_matrix.T /normalizer[:,None]
	importance = np.ones(num_teams)/num_teams
	zero_indices = np.where(~H.any(axis=1))[0] # check for any dangling nodes
	H[zero_indices,:] = importance # make H hat
	t = 0.85
	G = t * H + (1-t)/num_teams*np.ones((num_teams, num_teams))
	importance1 = converger(importance, G)
	importance_rank1 = np.argsort(importance1)[::-1]
	all_rankings[1,:] = importance_rank1

	#------------------------------------------------------------------------
	# Two more ways of ranking are used (deviants of Xia and Jain):
	# First: take the average based on number of games played in pairs
	# Second: take the total number of wins (this is similar to the baseline 
	# method described in Xia and Jain, but not clamped to 0 or 1)
	#------------------------------------------------------------------------
	baseline_matrix = np.zeros((num_teams, num_teams))
	for i in range(num_teams):
		for j in range(i+1, num_teams):
			winloss = teamvteam_np[i, j]
			winloss_split = winloss.split("-")
			if int(winloss_split[0]) > 0:
				relation_matrix[i, j] /= int(winloss_split[0])
				baseline_matrix[i, j] += int(winloss_split[0])
			if int(winloss_split[1]) > 0:
				relation_matrix[j, i] /= int(winloss_split[1])
				baseline_matrix[j, i] += int(winloss_split[1])
	# CURRENT WIN/LOSS STANDING RANKINGS
	curr_winloss = np.sum(baseline_matrix, axis=1)
	curr_standings = np.argsort(curr_winloss)[::-1]
	all_rankings[0,:] = curr_standings

	# FIRST METHOD
	# Run the PageRank Algorithm but for ranking teams, use AVERAGE!
	normalizer = np.sum(relation_matrix.T, axis=1)
	normalizer[normalizer == 0] = 1
	H = relation_matrix.T /normalizer[:,None]
	importance = np.ones(num_teams)/num_teams
	zero_indices = np.where(~H.any(axis=1))[0] # check for any dangling nodes
	H[zero_indices,:] = importance # make H hat
	t = 0.85
	G = t * H + (1-t)/num_teams*np.ones((num_teams, num_teams))
	importance2 = converger(importance, G)
	importance_rank2 = np.argsort(importance2)[::-1]
	all_rankings[2,:] = importance_rank2

	# SECOND METHOD
	normalizer = np.sum(baseline_matrix.T, axis=1)
	normalizer[normalizer == 0] = 1
	H = baseline_matrix.T / normalizer[:,None]
	importance = np.ones(num_teams)/num_teams
	zero_indices = np.where(~H.any(axis=1))[0] # check for any dangling nodes
	H[zero_indices,:] = importance # make H hat
	t = 0.85
	G = t * H + (1-t)/num_teams*np.ones((num_teams, num_teams))
	importance3 = converger(importance, G)
	importance_rank3 = np.argsort(importance3)[::-1]
	all_rankings[3,:] = importance_rank3

	#------------------------------------------------------------------------
	# PSYS-1 and PSYS-2
	#------------------------------------------------------------------------
	# modify which ranking system to default to
	ranking_list = [curr_standings, importance_rank1, importance_rank2, importance_rank3]
	for k in range(len(ranking_list)):
		ranking_system = ranking_list[k]
		# PSYS-1
		psys1_matrix = np.zeros((num_teams, num_teams))
		for i in range(num_teams):
			rank_i = np.where(ranking_system==i)[0][0]
			for j in range(i+1, num_teams):
				rank_j = np.where(ranking_system==j)[0][0]
				winloss = teamvteam_np[i, j]
				winloss_split = winloss.split("-")
				scaling = 1-((rank_i-rank_j)/num_teams)**2
				scaling_underdog = 1+((rank_i-rank_j)/num_teams)**2
				if int(winloss_split[0]) > 0:
					score = int(winloss_split[0])
					psys1_matrix[i, j] += score*scaling if rank_i > rank_j else score*scaling_underdog
				if int(winloss_split[1]) > 0:
					score = int(winloss_split[1])
					psys1_matrix[j, i] += score*scaling if rank_i < rank_j else score*scaling_underdog

		normalizer = np.sum(psys1_matrix.T, axis=1)
		normalizer[normalizer == 0] = 1
		H = psys1_matrix.T / normalizer[:,None]
		importance = np.ones(num_teams)/num_teams
		zero_indices = np.where(~H.any(axis=1))[0] # check for any dangling nodes
		H[zero_indices,:] = importance # make H hat
		t = 0.85
		G = t * H + (1-t)/num_teams*np.ones((num_teams, num_teams))
		importance = converger(importance, G)
		importance_rank = np.argsort(importance)[::-1]
		all_rankings[4+2*k,:] = importance_rank

		# PSYS-2
		psys2_matrix = np.zeros((num_teams, num_teams))
		for i in range(df.shape[0]):
			visit_ind = ind_dict[name_dict[visit_teams[i]]]
			home_ind = ind_dict[name_dict[home_teams[i]]]
			visit_rank = np.where(ranking_system==visit_ind)[0][0]
			home_rank = np.where(ranking_system==home_ind)[0][0]
			scaling = 1-((visit_rank-home_rank)/num_teams)**2
			scaling_underdog = 1+((visit_rank-home_rank)/num_teams)**2
			if diff_points[i] > 0:
				psys2_matrix[visit_ind, home_ind] += diff_points[i]*scaling if visit_rank > home_rank else diff_points[i]*scaling_underdog
			else:
				psys2_matrix[home_ind, visit_ind] += -diff_points[i]*scaling if visit_rank < home_rank else -diff_points[i]*scaling_underdog

		normalizer = np.sum(psys2_matrix.T, axis=1)
		normalizer[normalizer == 0] = 1
		H = psys2_matrix.T / normalizer[:,None]
		importance = np.ones(num_teams)/num_teams
		zero_indices = np.where(~H.any(axis=1))[0] # check for any dangling nodes
		H[zero_indices,:] = importance # make H hat
		t = 0.85
		G = t * H + (1-t)/num_teams*np.ones((num_teams, num_teams))
		importance = converger(importance, G)
		importance_rank = np.argsort(importance)[::-1]
		all_rankings[4+2*k+1,:] = importance_rank

#------------------------------------------------------------------------
# Use April as a test set and see the performance using the metrics
#------------------------------------------------------------------------
	test_game = all_games.iloc[test_num]
	test_visit = test_game[visit_name]
	test_home = test_game[home_name]
	test_diff = test_game[visit_pointcol]-test_game[home_pointcol]
	visit_win = test_diff > 0
	visit_ind = ind_dict[name_dict[test_visit]]
	home_ind = ind_dict[name_dict[test_home]]
	for j in range(all_rankings.shape[0]):
		ranking_system = all_rankings[j,:]
		visit_rank = np.where(ranking_system==visit_ind)[0][0]
		home_rank = np.where(ranking_system==home_ind)[0][0]
		visit_higher = visit_rank < home_rank
		# did it predict correctly?
		if visit_win == visit_higher:
			seq_acc[j] += 1
	all_acc[test_num-min_start,:]=seq_acc/(test_num+1-min_start)


# draw the graphs of these values over time
t = np.arange(min_start,all_games_num)
fig, (ax1) = plt.subplots(nrows = 1, ncols = 1)

label_list = ['NBA W/L','Weighted','APD','DB','NBA W/L+PSYS1','NBA W/L+PSYS2','Weighted+PSYS1','Weighted+PSYS2','APD+PSYS1','APD+PSYS2','DB+PSYS1','DB+PSYS2']
# plot for transmit power levels
lines = ax1.plot(t, all_acc)
ax1.set_title('Proportional Accuracy of Models')
ax1.legend(lines, label_list, loc='lower right', bbox_to_anchor=(1.1, 0))
ax1.set_ylabel('Proportion correct')
ax1.set_xlabel('Game number')
plt.show()