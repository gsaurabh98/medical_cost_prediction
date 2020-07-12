

import streamlit as st
import pandas as pd
import numpy as np
import json

import random

if not hasattr(st, 'already_started_server'):
    # Hack the fact that Python modules (like st) only load once to
    # keep track of whether this file already ran.
    st.already_started_server = True

    st.write('''
        The first time this script executes it will run forever because it's
        running a Flask server.

        Just close this browser tab and open a new one to see your Streamlit
        app.
    ''')

    from flask import Flask

    app = Flask(__name__)

    @app.route('/foo')
    def serve_foo():
        return 'This page is served via Flask!'

    app.run(port=5005)


# We'll never reach this part of the code the first time this file executes!

# Your normal Streamlit app goes here:

st.title('Medical cost prediction')

# st.subheader('Individual medical charges billed by health insurance')

st.markdown('<span><i>Individual medical charges billed by health insurance</i></span>',unsafe_allow_html=True)


st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.text('')
age = st.slider('Age: age of primary beneficiary', min_value =18, max_value=70)

st.text('')
sex= st.radio(
    "Sex: insurance contractor gender",
    ("male", "female")
)
st.text('')
bmi = st.number_input('Bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9', value= 18.5, key = 'bmi')

st.text('')
children = st.slider('Children: Number of children covered by health insurance / Number of dependents', min_value =0, max_value=10)
# st.number_input('Children', key = 'children')

st.text('')
smoker = st.radio(
	"Smoker: Smoking yes or no",
	("yes", "no")
	)


st.text('')
region = st.selectbox(
    "Region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.",
    ("southwest", "southeast", "northwest", "northeast")
)

#'predicted_value(charges)': bmi*20 
df = pd.DataFrame([{'age':age, 'sex':sex, 'bmi':bmi, 'children':children,'smoker':smoker,'region':region}])

st.text('')
with open('config.json', 'r') as f:
    config = json.load(f)

selected_model = st.selectbox(
    "Select Model",
    tuple(config['models'])
    )

pipeline = pd.read_pickle(config['pipeline'])

model = pd.read_pickle(config['models'][selected_model])


def highlight_green(series):
	color = 'green' if series.name == 'predicted_value(charges)' else 'black'
	return ['color: %s' % color]

st.text('')
if st.button('Predict'):
    X = pipeline.transform(df)
    df.insert(0, 'predicted_value(charges)', model.predict(X))
    st.text('')
    st.table(df.style.apply(highlight_green, axis=0))


# import time
# my_bar = st.progress(0)
# for percent_complete in range(100):
# 	time.sleep(0.1)
# 	my_bar.progress(percent_complete + 1)

# df = pd.read_csv('/home/winjit/Documents/predictsense/dataset/classification/insurance.csv')

# st.table(df)


# import streamlit as st
# import plotly.figure_factory as ff
# import numpy as np

# # Group data together
# hist_data = [df['charges']]

# group_labels = ['charges']
# # Create distplot with custom bin_size
# fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .25, .5])
# # Plot!
# st.plotly_chart(fig, use_container_width=True)