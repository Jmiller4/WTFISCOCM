import streamlit as st
import pandas as pd
import random
import numpy as np
from itertools import combinations
import altair as alt
from math import sqrt

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


def standard_qf(donation_df):
  projects = donation_df.columns
  funding = {p: (donation_df[p].apply(lambda x: sqrt(x)).sum() ** 2) - donation_df[p].sum() for p in projects}

  return funding


st.set_page_config(

		page_title="WTFISCOCM?",
		
)


st.write('## COCM rewards quality over influence')


s2 = 'At Gitcoin, we noticed a pattern: the projects that felt best for the ecosystem often had diverse bases of support, from engaged community members who donated to many other projects. '
s3 = 'Meanwhile, less legit-feeling projects sometimes made off with more funding because they were able to draw in many single or few-issue voters, either via bots or the promise of kickbacks and airdrops. '
s4 = 'This led to a theory about quality and influence: while normally correlated, sometimes projects with good quality have low influence, and vice versa. '
s5 = 'If quality and influence are correlated, then the best project wins in any funding regime. '
s6 = 'But when these two properties are mis-matched, **QF rewards the more influntial projects, and COCM rewards the projects with higher quality.**'

st.write(s2 + s3)
st.write(s4 + s5 + s6)

st.write('On this page, you can play around with a simulation comparing COCM and QF results when project quality and project influence are de-coupled.')
st.write('Use the sliders below to adjust the relative quality and influence of two main projects, A and B. Results for both QF and COCM are calculated below. Click the "advanced" tab to learn more, and remember that this is just a simple model of the world based on observations from Gitcoin -- your round or ecosystem may not behave like this.')


### ADVANCED DROP DOWN ###

exp = st.expander('Advanced Settings & In-depth Explanation')
exp.write('In this simulation, there is pool of core community members who will donate to both A and B proportional to their quality, and randomly donate to other projects in the round as well. There is also a pool of visiting community members, which is split into "A voters" and "B voters" proportional to the influence of each project. Visiting community members will donate mostly to their preferred project, and will donate randomly to other projects in the round, but with a smaller probability.')
exp.write('You can thnk of visiting community members as sybils, users mostly interested in an airdrop or other reward from donating, or users provoked to donate through a call to action.')
num_community_members = exp.slider('Number of core community members', value=500, min_value=50, max_value=1000, step=1)
num_call_to_actioners = exp.slider('Number of visiting community members', value=400, min_value=50, max_value=1000, step=1)
num_other_projects = exp.slider('Number of other projects', value=5, min_value=2, max_value=15, step=1)
cash_per_donor = exp.slider('Amount spent by each donor', value=10, min_value=5, max_value=30, step=1)


### QUALITY AND BALANCE SLIDERS AND CHARTS ###

col1, col2 = st.columns(2)

qual = col1.slider('Quality Balance', value = 0.2, min_value = 0.0, max_value=0.5, step = 0.01, label_visibility="collapsed")


source_qual = pd.DataFrame({
    'Project': ['A', 'B'],
    'Quality': [1-qual, qual]
})


cq = alt.Chart(source_qual, title = 'Quality Balance').mark_bar(width = 100).encode(
	alt.Y('Quality').scale(domain=(0, 1)),
    x='Project'
    
).properties(width=350, height = 300).configure_mark(color='green')

col1.altair_chart(cq)

inf = col2.slider('Influence Balance', value = 0.60, min_value = 0.5, max_value=1.0, step = 0.01, label_visibility="collapsed")

source_inf = pd.DataFrame({
    'Project': ['A', 'B'],
    'Influence': [1-inf, inf]
})

ci = alt.Chart(source_inf, title = 'Influence Balance').mark_bar(width = 100).encode(
	alt.Y('Influence').scale(domain=(0, 1)),
    x='Project'
    
).properties(width=350, height = 300).configure_mark(color='red')

col2.altair_chart(ci)


### RESULTS CALC + DISPLAY ###

def adjective(x):
	x = 1-x
	adj = 'much less'
	if x > 0.2: adj = 'less'
	if x > 0.4: adj = 'a little less'
	if x == 0.5: adj = 'the same'
	if x > 0.5: adj = 'a little more'
	if x > 0.6: adj = 'more'
	if x > 0.8: adj = 'much more'
	return adj

def preposition(x):
	prep = 'than'
	if x == 0.5: prep = 'as'
	return prep

conjunction = 'and'
if (qual - 0.5 > 0) != (inf - 0.5 > 0):
	conjunction = 'but'


st.write(f'#### Project A has {adjective(qual)} quality {preposition(qual)} project B, {conjunction} {adjective(inf)} influence.')

donor_profiles = []

for donor in range(num_community_members):
	random.seed(donor)
	rawbits = random.getrandbits(num_other_projects)
	bit_dict = {i: (rawbits >> i) & 1 for i in range(num_other_projects)}
	total_otherprojects = sum(bit_dict[i] for i in range(num_other_projects))
	total_mass = total_otherprojects * 0.5 + 1
	cash_per_otherproj = 0.5 / total_mass
	cash_for_A = (1-qual) / total_mass
	cash_for_B = qual / total_mass

	don_dict = {0: cash_for_A, 1:cash_for_B}
	for p in range(num_other_projects):
		if bit_dict[p]:
			don_dict[p+2] = cash_per_otherproj
		else:
			don_dict[p+2] = 0
	donor_profiles.append(pd.Series(don_dict))


for donor in range(num_call_to_actioners):
	### TODO: these people contribute to other projects as well with some smaller probability
	don_dict = {i: 0 for i in range(num_other_projects+2)}

	remaining_cash = cash_per_donor

	for p in range(num_other_projects):
		random.seed(donor * 2000 + num_other_projects)
		if random.random() < 0.15:
			don_dict[p + 2] = cash_per_donor / (num_other_projects + 1)
			remaining_cash -= cash_per_donor / (num_other_projects + 1)

	if donor < (1 - inf) * num_call_to_actioners:
		don_dict[0] = remaining_cash
	else:
		don_dict[1] = remaining_cash

	
	donor_profiles.append(pd.Series(don_dict))


don_df = pd.concat(donor_profiles,axis=1).transpose()

COCM_res = COCM(don_df, don_df)

QF_res = standard_qf(don_df)

# st.write(convert_to_pcts(QF_res))
# st.write(convert_to_pcts(COCM_res))


if QF_res[0] + QF_res[1] > 0:
	QF_A_bal = QF_res[1] / (QF_res[0] + QF_res[1])

	col = 'black'
	if 1 - QF_A_bal > 0.5:
		col = 'green'
	if 1 - QF_A_bal < 0.5:
		col = 'red'
	st.write(f'#### Under QF, project A gets :{col}[{adjective(QF_A_bal)}] funding {preposition(QF_A_bal)} project B.')
else:
	st.write(f'#### Under QF, neither project gets any funding.')


if COCM_res[0] + COCM_res[1] > 0:
	COCM_A_bal = COCM_res[1] / (COCM_res[0] + COCM_res[1])
	col = 'black'
	if 1 - COCM_A_bal > 0.5:
		col = 'green'
	if 1 - COCM_A_bal < 0.5:
		col = 'red'
	s = f'#### Under COCM, project A gets :{col}[{adjective(COCM_A_bal)}] funding {preposition(COCM_A_bal)} project B.'
	st.write(s)
else:
	st.write(f'#### Under QF, neither project gets any funding.')


res_col1, res_col2 = st.columns(2)


QF_res_src = pd.DataFrame({
    'Project': ['A', 'B'],
    'Funding share (QF)': [1 - QF_A_bal, QF_A_bal]
})


cQF = alt.Chart(QF_res_src, title = 'QF results').mark_bar(width = 100).encode(
	alt.Y('Funding share (QF)').scale(domain=(0, 1)),
    x='Project'
    
).properties(width=350, height = 300).configure_mark(color='orange')

res_col1.altair_chart(cQF)


COCM_res_src = pd.DataFrame({
    'Project': ['A', 'B'],
    'Funding share (COCM)': [1 - COCM_A_bal, COCM_A_bal]
})

cCOCM = alt.Chart(COCM_res_src, title = 'COCM results').mark_bar(width = 100).encode(
	alt.Y('Funding share (COCM)').scale(domain=(0, 1)),
    x='Project'
    
).properties(width=350 , height = 300).configure_mark(color='blue')

res_col2.altair_chart(cCOCM)



results_df = pd.DataFrame(columns = ['Funding share - Standard QF', 'Funding share - COCM'], index= ['Project A', 'Project B', f'{num_other_projects} other projects'], data=0)

QF_pcts = convert_to_pcts(QF_res)
COCM_pcts = convert_to_pcts(COCM_res)

def stringify(x):
	if x >= 1:
		return '100%'
	if x >= 0.1:
		return str(x)[2:4] + '%'
	return str(x)[3] + '%'

results_df.loc['Project A', 'Funding share - Standard QF'] = stringify(QF_pcts[0])
results_df.loc['Project B', 'Funding share - Standard QF'] = stringify(QF_pcts[1])
results_df.loc[f'{num_other_projects} other projects', 'Funding share - Standard QF'] = stringify(sum(QF_pcts[i+2] for i in range(num_other_projects)))
results_df.loc['Project A', 'Funding share - COCM'] = stringify(COCM_pcts[0])
results_df.loc['Project B', 'Funding share - COCM'] = stringify(COCM_pcts[1])
results_df.loc[f'{num_other_projects} other projects', 'Funding share - COCM'] = stringify(sum(COCM_pcts[i+2] for i in range(num_other_projects)))
st.write('#### Full results:')
st.write(results_df)
