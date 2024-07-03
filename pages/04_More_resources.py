import streamlit as st

st.set_page_config(

		page_title="WTFISCOCM?",
		
)

st.write('## More Resources')


BCR_url = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4311507"
WTF_url = "https://www.gitcoin.co/blog/wtf-is-cluster-matching-qf"
Forum_url = "https://gov.gitcoin.co/t/nerd-post-updates-to-cluster-mapping-matching/18705"
COCM_prod_url = "https://github.com/Jmiller4/qf-variants"
COCM_illus_url = "https://colab.research.google.com/drive/1D_3h-Lvpvd5p2YUbyhV8pMRzCZmbFxbk?usp=sharing"
Sandbox_url = "https://github.com/Jmiller4/COCM-sandbox"

st.write(f'The first iteration of COCM was described in [this paper]({BCR_url}). The paper also includes a discussion of collusion and QF\'s weakness to collusion.')

st.write(f'[This short Gitcoin post]({WTF_url}) uses cool visuals to illustrate the ideas behind the first iteration COCM.')

st.write(f'Since then, the version of COCM used in production at Gitcoin (and on this webpage) has undergone some tweaks. [This forum post]({Forum_url}) describes those tweaks in detail.')

st.write(f'The production version of COCM used at Gitcoin is available [in this repository]({COCM_prod_url}). Easier to read (but much slower) code illustrating the ideas behind the production version of COCM is available [in this notebook]({COCM_illus_url}). The code for this streamlit app is hosted [here]({Sandbox_url}).')
