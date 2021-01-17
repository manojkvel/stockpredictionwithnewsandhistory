from pandas import datetime

import helper
import numpy as np
import pandas as pd
import yahoo_finance as yf


def perform_empirical_analysis(ticker, freq, polarities):
    yf.download_from_yahoo_finance([ticker], "2020-11-24",
                                   datetime.today().strftime('%Y-%m-%d'))
    file_name = ticker + "_" + freq + ".csv"
    dataset = pd.read_csv(file_name)

    if type(polarities) == str:
        stocks = helper.itemize_emp_data(dataset)
    else:
        stocks = helper.itemize_pol_data(dataset, polarities)

    stocks = helper.get_normalised_data(stocks)
    print(stocks.head())

    print("\n")
    print("Open   --- mean :", np.mean(stocks['Open']), "  \t Std: ", np.std(stocks['Open']), "  \t Max: ",
          np.max(stocks['Open']), "  \t Min: ", np.min(stocks['Open']))
    print("Close  --- mean :", np.mean(stocks['Close']), "  \t Std: ", np.std(stocks['Close']), "  \t Max: ",
          np.max(stocks['Close']), "  \t Min: ", np.min(stocks['Close']))
    print("Volume --- mean :", np.mean(stocks['Volume']), "  \t Std: ", np.std(stocks['Volume']), "  \t Max: ",
          np.max(stocks['Volume']), "  \t Min: ", np.min(stocks['Volume']))
    if type(polarities) != str:
        print("Polarity --- mean :", np.mean(stocks['Polarity']), "  \t Std: ", np.std(stocks['Polarity']),
              "  \t Max: ",
              np.max(stocks['Polarity']), "  \t Min: ", np.min(stocks['Polarity']))

    # Print the dataframe head and tail
    print(stocks.head())
    print("---")
    print(stocks.tail())

    return stocks
