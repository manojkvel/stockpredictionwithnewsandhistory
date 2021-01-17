import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['figure.figsize'] = (18, 12)


def price(x):
    """
    format the coords message box
    :param x: data to be formatted
    :return: formatted data
    """
    return '$%1.2f' % x


def get_normalised_data(data):
    """

    :param data: dataset to be normalized by the MinMaxScaler()
    :return:
    """
    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler()
    numerical = ['Open', 'Close', 'Volume']
    data[numerical] = scaler.fit_transform(data[numerical])

    return data


def itemize_pol_data(input, polarities):
    """

    :param input: stock price history
    :param polarities: polarities to be merged with stock prices
    :return:
    """
    input['Datetime'] = pd.to_datetime(input['Datetime'])
    input['Datetime'] = input['Datetime'].dt.tz_localize(None)
    data = pd.merge(input, polarities, on="Datetime", how="outer").fillna(0)
    data = data.loc[data['Open'] * data['Close'] != 0]
    # Define columns of data to keep from historical stock data
    item = []
    open = []
    close = []
    volume = []
    polarity = []

    # Loop through the stock data objects backwards and store factors we want to keep
    i_counter = 0
    for i in range(len(data) - 1, -1, -1):
        item.append(i_counter)
        open.append(data['Open'][i])
        close.append(data['Close'][i])
        volume.append(data['Volume'][i])
        polarity.append(data['Polarity'][i])
        i_counter += 1

    # Create a data frame for stock data
    stocks = pd.DataFrame()

    # Add factors to data frame
    stocks['Item'] = item
    stocks['Open'] = open
    stocks['Close'] = pd.to_numeric(close)
    stocks['Volume'] = pd.to_numeric(volume)
    stocks['Polarity'] = pd.to_numeric(polarity)

    # return new formatted data
    return stocks


def itemize_emp_data(data):
    """

    :param data: dataset to itemize
    :return:
    """
    # Define columns of data to keep from historical stock data
    item = []
    open = []
    close = []
    volume = []

    # Loop through the stock data objects backwards and store factors we want to keep
    i_counter = 0
    for i in range(len(data) - 1, -1, -1):
        item.append(i_counter)
        open.append(data['Open'][i])
        close.append(data['Close'][i])
        volume.append(data['Volume'][i])

        i_counter += 1

    # Create a data frame for stock data
    stocks = pd.DataFrame()

    # Add factors to data frame
    stocks['Item'] = item
    stocks['Open'] = open
    stocks['Close'] = pd.to_numeric(close)
    stocks['Volume'] = pd.to_numeric(volume)

    # return new formatted data
    return stocks


def plot_basic(stocks, title='Empirical Data', y_label='Price USD', x_label='Trading Days'):
    """

    :param stocks: Stock price history
    :param title: Plot title
    :param y_label: y label of the plot
    :param x_label: c label of the plot
    """
    fig, ax = plt.subplots()
    ax.plot(stocks['Item'], stocks['Close'], '#0A7388')

    ax.format_ydata = price
    ax.set_title(title)

    # Add labels
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    plt.show()


def plot_lstm_prediction(actual, prediction, chart_name='Plot.png', title='Actual vs Prediction', y_label='Price USD',
                         x_label='Trading Days'):
    """
    Plots train, test and prediction
    :param chart_name: Name of the chart being plotted
    :param actual: DataFrame containing actual data
    :param prediction: DataFrame containing predicted values
    :param title:  Title of the plot
    :param y_label: yLabel of the plot
    :param x_label: xLabel of the plot
    :return: prints a Pyplot againts items and their closing value
    """
    title = chart_name
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Add labels
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    # Plot actual and predicted close values

    plt.plot(actual, '#00FF00', label='Adjusted Close')
    plt.plot(prediction, '#0000FF', label='Predicted Close')

    # Set title
    ax.set_title(title)
    ax.legend(loc='upper left')

    plt.savefig(chart_name, bbox_inches='tight')
