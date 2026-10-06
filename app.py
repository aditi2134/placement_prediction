import streamlit as st
import pandas as pd 
import joblib as jb 

model = jb.load('logisticregression.pkl')

st.title('PLACEMENT PREDICTION')
st.markdown("Provide the details")

cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=6.0, step=0.1)

iq = st.number_input("IQ", min_value=50, max_value=160, value=100, step=1)

if st.button("Predict"):
    input_df = pd.DataFrame({
        'cgpa': [cgpa],
        'iq': [iq]
    })

    prediction = model.predict(input_df)

    st.write(prediction)