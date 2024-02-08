import streamlit as st
import pandas as pd

st.write(
    """
# predictor
"""
)

# df = pd.read_csv("my_data.csv")
# st.line_chart(df)

st.button("click me")

import numpy as np
import Vision.Actiondetect as action
# import Vision.graph as graph


batch = np.load("./src/Vision/testarr.npy")

st.write(batch.shape)
st.write(
    """
module loaded!
"""
)

def fetch():
    predictions = action.predict_action(batch)
    st.write(predictions)

if st.button("Make prediction"):  # data is hidden if box is unchecked
    fetch()
    st.image('src/Vision/one.png' ,use_column_width='auto')



# st.write(predictions)

# if st.checkbox("Show raw data"):  # data is hidden if box is unchecked
#     st.write(df)

