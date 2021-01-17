import yfinance as yf
from pandas_datareader import data as pdr


def download_from_yahoo_finance(tickers, start_date, end_date):
    yf.pdr_override()

    for i in tickers:
        data = pdr.get_data_yahoo(i, start_date, end_date, interval='30m')
        data.to_csv(i + '_30m.csv')
    #   data.DataReader(i,'yahoo',start_date,end_date).to_csv(i+'.csv')
