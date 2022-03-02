import numpy as np
import pandas as pd
import yfinance as yf
import datetime 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import arch, pmdarima
import streamlit as st
from stock import Stock
from arch.__future__ import reindexing

def train(df):
	x = df['Close']
	history = x
	df_new = pd.DataFrame(history)
	df_new['Close_New'] = history.iloc[:int(0.8*len(x))]
	df_new['Close_Pred'] = history.iloc[:int(0.8*len(x))]
	arima = pmdarima.ARIMA(order = (1,0,1))
	for i in range(int(0.8*len(x)), len(x)):
		arima.fit(df_new.iloc[0:i]['Close'])
		pred = arima.predict(n_periods = 1)[0]

		garch = arch.arch_model(arima.resid())
		garch_fit = garch.fit(disp='off', show_warning=False)
		garch_pred = garch_fit.forecast(horizon=1).mean.iloc[-1]['h.1']
		df_new['Close_New'][df.index[i]] = pred + garch_pred

		arima.fit(df_new.iloc[0:i]['Close_Pred'])
		pred = arima.predict(n_periods=1)[0]

		garch = arch.arch_model(arima.resid())
		garch_fit = garch.fit(disp='off', show_warning=False)
		garch_pred = garch_fit.forecast(horizon=1).mean.iloc[-1]['h.1']
		df_new['Close_Pred'][df.index[i]] = pred + garch_pred
	return df_new

def best_order(df):
	x = df['Close']
	arima = pmdarima.ARIMA(order = (0,0,0))
	best_cost = np.inf
	best_tuple = (0,0,0)
	for i in range(5):
		for j in range(5):
			arima.order = (i,0,j)
			arima.fit(x)
			if best_cost > arima.aic():
				best_tuple = (i,0,j)
	return best_tuple

def future_predict(df):
	history = df['Close']
	arima = pmdarima.ARIMA(order = best_order(df))
	for i in range(14):
		arima.fit(history)
		pred = arima.predict(n_periods = 1)[0]
		garch = arch.arch_model(arima.resid())
		garch_fit = garch.fit(disp='off', show_warning = False)
		garch_pred = garch_fit.forecast(horizon=1).mean.iloc[-1]['h.1']

		history[history.index[-1] + datetime.timedelta(days=1)] = pred + garch_pred
	return history

def final_printing(history):

	std_dev = np.std(history)
	avg = np.mean(history)
	inc1 = np.max(history) - history[0]
	dec1 = history[0] - np.min(history)

	if(inc1 > dec1):
		if(inc1 > std_dev):
			print('Significant increase, BUY')
		else:
			print('Slight increase, HOLD')
	else:
		if(dec1 > std_dev):
			print('Significant decrease, SHORT')
		else:
			print('Slight Decrease, HOLD')
	st.write('average: '+str(avg))
	st.write('STD: '+str(std_dev))
	st.write('Minimum: '+str(np.min(history)))
	st.write('Maximum: '+str(np.max(history)))
	st.write('Start: '+str(history[0]))
	return


if(__name__=='__main__'):
	stock_ops = ('UBER', 'TSLA', 'SENSEX', 'NIFTY_50', 'AMZN')
	st.title('This is the thing')

	stock_option = st.selectbox('Choose Stock Option', stock_ops)

	st.write('You selected:',stock_option)

	dates = st.slider('Dates to choose for further prediction', value = (datetime.datetime(2019, 1, 1),datetime.datetime(2020, 1, 1)), format = "DD/MM/YY")
	df_stock = yf.download(tickers = stock_option, period = '100000d', interval = '1d')

	df = df_stock[df_stock.index >= dates[0]]
	df = df[df.index <= dates[1]]	
	fig = go.Figure()
	fig.add_trace(go.Candlestick(x = df.index, open = df['Open'], low = df['Low'], high = df['High'], close = df['Close'], name = 'Stock Market Data for '+stock_option))
	fig.update_layout(xaxis_rangeslider_visible=False)
	st.write(fig)

	st.write('sdfsfsdfsdf')

	df_new = train(df)
	st.write(plt.plot(df_new))

	history = future_predict(df_new)
	fig, ax = plt.subplots()
	ax.plot(history.iloc[:-15], marker = '.')
	ax.plot(history.iloc[-15:], marker = '.', linestyle = '--')
	st.write(fig)

	final_printing(np.array(history.iloc[-15:]))