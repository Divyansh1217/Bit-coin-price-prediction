import pandas as pd
from prophet import Prophet
import streamlit as st
data=pd .read_csv("BTC-USD.csv")
df=data[["Date","Close"]]
df.columns=["ds","y"]
model=Prophet()
model.fit(df)
future=model.make_future_dataframe(periods=365)
fcst=model.predict(future)
st.write(fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]])
fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(200)
from prophet.plot import plot
st.pyplot(model.plot(fcst=fcst,figsize=(20,20)))


