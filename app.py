# Imports
import streamlit as st
import pandas as pd
import pickle

# Opening intro text
st.title("Reducing NBA Injuries")

st.markdown('Season ending injuries are some of the most devastating for NBA players. \
            I created this tool to identify NBA players at high risk for season-ending injuries. \
            Select a player and discover their current season predicted probability of season-ending injury.')

players = pickle.load(open('CurrentPlayers.sav', 'rb'))

name = st.selectbox(
    'Select a player...',
    options = players)

# st.write(name)

# Return the row of the dataframe with the selected name
row = players.loc[players.Name == name].iloc[0]

# Convert row to standalone dataframe
player = players.loc[players.Name == name].iloc[0,1:].to_frame().T

# st.write(player)

# Load model
model = pickle.load(open('Model.sav', 'rb'))

# Run predictions
pred = model.predict(player)[0]
proba = model.predict_proba(player)[0][1]

# st.write(pred)
# st.write(proba)

# Sharing the predictions
if pred == 1:
    st.write(f"## {name} is at :red[high risk] of season-ending injury!")
    st.write(f"Predicted probability of season-ending injury: {proba*100:.2f} %")

else:
    st.write(f"## {name} is not at high risk of season-ending injury!")
    st.write(f"Predicted probability of season-ending injury: {proba*100:.2f} %")