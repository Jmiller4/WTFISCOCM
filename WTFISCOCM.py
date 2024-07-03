import streamlit as st

st.set_page_config(

		page_title="WTFISCOCM?",
		
)

st.image('657c7ed16b14af693c08b92d_GTC-Logotype-Dark.png', width = 300)


st.write('# WTF IS COCM?')

st.write('COCM (Connection-Oriented Cluster Match) is a mechanism for allocating money to different projects. Users donate money directly to the projects, and then the mechanism adds in extra money from a "matching pool" to award projects that have diverse bases of support.')

st.write('COCM builds off of another mechanism called Quadratic Funding (QF). QF is great, but it\'s not so good at dealing with fake accounts or well-coordinated groups of people, which can sometimes make it feel like a popularity contest where the project with the most influence win, rather than the projects with the most quality. COCM fixes the problems with QF by refining where and when projects get extra money.')

st.write('This page contains two tools for exploring COCM. In the first, you can run a simulation to how COCM reduces the amount that projects can influence results through airdrops, sybils, and calls to action. On the second page, you can play around with an exact configuration of donations and see what happens.')

st.write('## Should I use COCM in my funding round?')
st.write('Every mechanism has strengths and weaknesses. COCM is good at **prioritizing the voices of real people and core community members, or any donor who takes the time to craft a unique slate of donations**. However, COCM also **reduces the impact that single-issue voters have**, and **reduces the impact that groups of similar voters have**. Sometimes, single issue voters and voters with similar donation patterns constitute the newest members of a community, so COCM is not ideal for prioritizing those voices.')