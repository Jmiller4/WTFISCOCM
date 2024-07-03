import streamlit as st
import pandas as pd
import random
import numpy as np
from itertools import combinations


def binarize(df):
	return df.applymap(lambda x: 1 if x > 0 else 0)

def convert_to_pcts(funding_dict):
	total = sum(funding_dict[k] for k in funding_dict.keys())
	return {k : funding_dict[k] / total for k in funding_dict.keys()}


def drop_empty_projects(donation_df):

	for c in donation_df.columns:
		if donation_df[c].sum() == 0:
			donation_df.drop(c, inplace=True, axis=1)
	return donation_df


def COCM(donation_df, cluster_df, calcstyle='markov', harsh=True):
	# run CO-CM on a set of funding amounts and clusters
	# calcstyle is a variable signifying how to compute similarity scores between users and projects
	# harsh is a boolean determining how we should scale contributions, given similarity scores
	projects = donation_df.columns
	clusters = cluster_df.columns
	donors = donation_df.index
	cluster_members = cluster_df.index

	# normalize the cluster dataframe so that rows sum to 1. Now, an entry tells us the "weight" that a particular cluster has for a particular user.
	# if a user is in 0 clusters, their row will be a bunch of NaNs if we naively divide by 1.
	# we shouldn't have any such users anyways, but just in case, we'll fill such a row with 0s instead
	normalized_clusters = cluster_df.apply(lambda row: row / row.sum() if any(row) else 0, axis=1)

	binarized_clusters = binarize(cluster_df)

	assert calcstyle in ['markov', 'og', 'pct_friends']
	if calcstyle == 'markov':
		normalized_users = cluster_df.transpose().apply(lambda row: row / row.sum() if any(row) else 0, axis=1)
		PJP = normalized_users.dot(normalized_clusters)
		k_indicators = normalized_clusters.dot(PJP)
		k_indicators = k_indicators.apply(lambda row: np.maximum(row, binarized_clusters.loc[row.name]), axis=1)

	if calcstyle == 'pct_friends':
		# friendship_matrix is a matrix whose rows and columns are both wallets,
		# and a value of 1 at index i,j means that wallets i and j are in at least one cluster together.
		friendship_matrix = cluster_df.dot(cluster_df.transpose()).apply(lambda col: col > 0)

		# k_indicators is a dataframe with wallets as rows and clusters as columns.
		# if wallet i is not in cluster g, then entry i,g is is the fraction of i's friends who are in cluster g (i's friends are the agents i is in a shared cluster with).
		# if wallet i is in cluster g, then entry i,g is 1.

		# in the past, we used cluster_df in the following line instead of binarized_clusters
		k_indicators = friendship_matrix.dot(binarized_clusters).apply(lambda row: row / friendship_matrix.loc[row.name].sum(), axis=1)
		# ... and the following line used cluster_df instead of binarized_clusters
		k_indicators = k_indicators.apply(lambda row: np.maximum(row, binarized_clusters.loc[row.name]), axis=1)
	
	if calcstyle == 'og':
		# friendship_matrix is a matrix whose rows and columns are both wallets,
		# and a value greater than 0 at index i,j means that wallets i and j are in at least one group together.
		friendship_matrix = cluster_df.dot(cluster_df.transpose())

		# k_indicators is a dataframe with wallets as rows and stamps as columns.
		# entry i,g is True if wallet i is in a shared group with anyone from g, and False otherwise.
		k_indicators = friendship_matrix.dot(cluster_df).apply(lambda col: col > 0)

	# Create a dictionary to store funding amounts for each project.
	funding = {p: 0 for p in projects}

	for p in projects:
		# get the actual k values for this project using contributions and indicators.

		# C will be used to build the matrix of k values.
		# It is a matrix where rows are wallets, columns are clusters, and the ith row of the matrix just has wallet i's contribution to the project in every entry.
		C = pd.DataFrame(index=donors, columns = ['_'], data = donation_df[p].values).dot(pd.DataFrame(index= ['_'], columns = clusters, data=1))
		# C is attained by taking the matrix multiplication of the column vector donation_df[p] (which is every agent's donation to project p) and a row vector with as many columns as projects, and a 1 in every entry
		# the above line is so long mainly because you need to cast Pandas series' (i.e. vectors) as dataframes (i.e. matrices) for the matrix multiplication to work.

		# now, K is a matrix where rows are wallets, columns are projects, and entry i,g ranges between c_i and sqrt(c_i) depending on i's relationship with cluster g and whether "fancy" was set to true or not.
		K = (k_indicators * C.pow(1/2)) + ((1 - k_indicators) * C)

		if harsh == True:
			K = (1 - k_indicators) * C

		# Now we have all the k values, which are one of the items inside the innermost sum expressed in COCM.
		# the other component of these sums is a division of each k value by the number of groups that user is in.
		# P_prime is a matrix that combines k values and total group memberships to attain the value inside the aforementioned innermost sum.
		# In other words, entry g,h of P_prime is:
		#
		#       sum_{i in g} K(i,h) / T_i
		#
		# where T_i is the total number of groups that i is in
		P_prime = K.transpose().dot(normalized_clusters)

		# Now, we can create P_prime, whose non-diagonal entries g,h represent the pairwise subsidy given to the pair of groups g and h.
		P = (P_prime * P_prime.transpose()).pow(1/2)

		# The diagonal entries of P are not relevant, so get rid of them. We only care about the pairwise subsidies between distinct groups.
		np.fill_diagonal(P.values, 0)

		# Now the sum of every entry in P is the amount of subsidy funding COCM awards to the project.
		funding[p] += P.sum().sum()


	return funding




def rand_matrix(w,h):
	
	M = pd.DataFrame(index = range(h), columns = range(w), data=0)

	M.loc[0,random.choice(list(range(w)))] = 1
	M.loc[0,random.choice([j for j in range(w) if M.loc[0,j] == 0])] = 1
	M.loc[0,random.choice([j for j in range(w) if M.loc[0,j] == 0])] = -1

	for j in range(w):
		if M.loc[0,j] == 0:
			M.loc[0,j] = random.choice([1,-1])


	for i in range(1,h):

		#assert len([j for j in range(n) if M[i-1][j] == 1]) >=2

		ones = [j for j in range(w) if M.loc[i-1,j] == 1]
		# st.write(f'row {i}: {ones}, {neg_ones}')
		# st.write(ones)
		neg_ones = [j for j in range(w) if M.loc[i-1,j] == -1]
		# st.write(neg_ones)

		pair = random.choice(list(combinations(ones,2)))


		if random.random() > 0.5:
			M.loc[i,pair[0]] = 1
			M.loc[i,pair[1]] = -1
		else:
			M.loc[i,pair[0]] = -1
			M.loc[i,pair[1]] = 1

		M.loc[i,random.choice(neg_ones)] = 1
		
		# st.write(f'row {i}: {ones}, {neg_ones}, {pair}')

		# the rest random
		for j in range(w):
			if M.loc[i,j] == 0:
				M.loc[i,j] = random.choice([1,-1])

	# st.write(M)
	return M


def randomize_donations(ld, lp):

	M = rand_matrix(lp, ld)

	for d in range(max_donors):
		for p in range(max_projects):
			st.session_state.def_don[d][p] = 0

			if d < ld and p < lp:
				if M.loc[d,p] == 1:
					st.session_state.def_don[d][p] = random.choice(list(range(1,max_donation+1)))

			# elif d in donors and p in projects:

			# 	if random.random() > 0.666:
			# 		st.session_state.def_don[d][p] = random.choice(list(range(1,max_donation+1)))


def clear_donations():
	for d in range(max_donors):
		for p in range(max_projects):
			st.session_state.def_don[d][p] = 0
			st.session_state.don[d][p] = 0

st.set_page_config(
		page_title="WTFISCOCM?",
		layout="wide",
)

st.write("# COCM Sandbox")

st.write('On this page, you can play around with COCM yourself. Use the sliders to adjust donation amounts. Final funding amounts are displayed at the bottom.')

st.write('**For a project to get extra funding (above what folks directly donate), the donor group needs to be diverse**: there needs to be at least two other projects (let\'s call them X and Y) where some people also donate to X and not Y, and other people do the opposite.')

st.write("### Configuration")




names = ['Alice','Bob','Cassie','Denice','Etna','Frankie','Gertie','Hal','Irena','June']
max_donors = len(names)
max_projects = 10
max_donation = 50


col1, col2, col3 = st.columns(3)
num_donors = col1.number_input("Number of Donors", min_value=3, max_value=max_donors, value=5, step = 1)
num_projects = col2.number_input("Number of Projects", min_value=3, max_value=max_projects, value=3, step = 1)
matching_pool_size = col3.slider('Size of Matching Pool', min_value=100, max_value=2000, step=100)


donors = range(num_donors)
projects = range(num_projects)

# if 'startup' not in st.session_state:
# 	randomize_donations(num_donors, num_projects)
# 	st.session_state.startup = 'done'


#st.write("### Donation Amounts")

if 'don' not in st.session_state:

	st.session_state.don = {d: {p:0 for p in range(max_projects)} for d in range(max_donors) }

if 'def_don' not in st.session_state:

	st.session_state.def_don = {d: {p:0 for p in range(max_projects)} for d in range(max_donors) }

if 'iter' not in st.session_state:
	st.session_state.iter = 0

# st.session_state.iter += 1



bc1, bc2 = st.columns(2)
if bc1.button('Shuffle Donations'):
	randomize_donations(num_donors, num_projects)
	st.session_state.iter += 1
if bc2.button('Clear Donations'):
	clear_donations()
	st.session_state.iter += 1



# render sliders


proj_cols = st.columns(num_projects)

for p in projects:
	for d in donors:
		#st.session_state['don'][d][p] = 
		# if clearing:
		# 	st.session_state.don[d][p] = 0
		# 	st.session_state.don[d][p] =  proj_cols[p].slider(f'{names[d]} ➡️ Project {p + 1}', min_value = 0, max_value = max_donation, value =0)
		# 	clearing = False
		# else:
		# 	st.session_state.don[d][p] = proj_cols[p].slider(f'{names[d]} ➡️ Project {p + 1}', min_value = 0, max_value = max_donation, value = st.session_state.def_don[d][p])
		
		slider_placeholder = proj_cols[p].empty()

		st.session_state.don[d][p] = proj_cols[p].slider(f'{names[d]} ➡️ Project {p + 1}', min_value = 0, max_value = max_donation, value = st.session_state.def_don[d][p], key=f'{d},{p},{st.session_state.iter}')



st.session_state.donation_df = pd.DataFrame(index = donors, columns = projects, data = [[st.session_state.don[d][p] for p in projects] for d in donors])


for d in donors:
	if st.session_state.donation_df.loc[d].sum() == 0:
		st.session_state.donation_df.drop(d, inplace=True)

projects_without_donors = []

if len(st.session_state.donation_df) == 0:
	res = {p: 0 for p in projects}
	projects_without_donors = list(projects)
else:
	for p in projects:
		if st.session_state.donation_df[p].sum() == 0:
			st.session_state.donation_df.drop(p, axis=1, inplace=True)
			projects_without_donors.append(p)

	if len(projects_without_donors) == num_projects:
		res = {p: 0 for p in projects}
	else:	
		res = COCM(st.session_state.donation_df , st.session_state.donation_df)

		for p in projects_without_donors:
			res[p] = 0

	# for p in st.session_state.donation_df.columns:
	# 	res[p] += st.session_state.donation_df[p].sum()


total_matching = sum(res[p] for p in projects)


st.write("### Results")

res_df = pd.DataFrame(index=[f'Project {p + 1}' for p in projects], columns = ['Donations from Users', 'Percent of Matching Pool', 'Total Funding'])

for p in projects:

	if total_matching > 0:
		res_df.loc[f'Project {p + 1}','Percent of Matching Pool'] = res[p] / total_matching
	else:
		res_df.loc[f'Project {p + 1}','Percent of Matching Pool'] = 0

	if p in projects_without_donors:
		res_df.loc[f'Project {p + 1}','Donations from Users'] = 0
		res_df.loc[f'Project {p + 1}','Total Funding'] = 0
	else:
		res_df.loc[f'Project {p + 1}','Donations from Users'] = st.session_state.donation_df[p].sum()
		res_df.loc[f'Project {p + 1}','Total Funding'] = res_df.loc[f'Project {p + 1}','Percent of Matching Pool'] * matching_pool_size + st.session_state.donation_df[p].sum()


st.table(res_df)

# st.table(res)
#asdfjewiofjsdfdfgsadfaewiofoij

