import helper
import io
import math
import time

import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils.vis_utils import plot_model

import empirical_analysis as ea
import sentiment_analysis as sa

output_map = {}


def unroll(data, sequence_length=24):
    """
    use different windows for testing and training to stop from leak of information in the data
    :param data: data set to be used for unrolling
    :param sequence_length: window length
    :return: data sets with different window.
    """
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    return np.asarray(result)


def lstm_model2(input_dim, output_dim, return_sequences):
    """
    Builds an improved Long Short term memory model using keras.layers.recurrent.lstm
    :param input_dim: input dimension of model
    :param output_dim: ouput dimension of model
    :param return_sequences: return sequence for the model
    :return: a 3 layered LSTM model
    """
    model = Sequential()
    model.add(LSTM(
        input_shape=(None, input_dim),
        units=output_dim,
        return_sequences=return_sequences))

    model.add(Dropout(0.2))

    model.add(LSTM(
        100,
        return_sequences=False))

    model.add(Dropout(0.2))

    model.add(Dense(
        units=1))
    model.add(Activation('linear'))

    return model


def lstm_model(input_dim, output_dim, return_sequences):
    """
    Builds a basic lstm model
    :param input_dim: input dimension of the model
    :param output_dim: output dimension of the model
    :param return_sequences: return sequence of the model
    :return: a basic lstm model with 3 layers.
    """
    model = Sequential()
    model.add(LSTM(
        input_shape=(None, input_dim),
        units=output_dim,
        return_sequences=return_sequences))

    model.add(LSTM(
        50,
        return_sequences=False))

    model.add(Dense(
        units=1))
    model.add(Activation('linear'))

    return model


def model_with_empirical_data(emp_df, tuned, chart_name):
    # training data
    # rounded_test_size = round(emp_df.size * .2).astype(int)
    test_data_cut = 100 + 10 + 1
    prediction_time = 1

    x_train = emp_df[0:-prediction_time - test_data_cut].values
    y_train = emp_df[prediction_time:-test_data_cut]['Close'].values

    # test data
    x_test = emp_df[0 - test_data_cut:-prediction_time].values
    y_test = emp_df[prediction_time - test_data_cut:]['Close'].values
    print("x_train", x_train.shape)
    print("y_train", y_train.shape)
    print("x_test", x_test.shape)
    print("y_test", y_test.shape)

    unroll_length = 5
    x_train = unroll(x_train, unroll_length)
    x_test = unroll(x_test, unroll_length)
    y_train = y_train[-x_train.shape[0]:]
    y_test = y_test[-x_test.shape[0]:]

    print("x_train", x_train.shape)
    print("y_train", y_train.shape)
    print("x_test", x_test.shape)
    print("y_test", y_test.shape)

    if tuned:
        model = lstm_model2(input_dim=x_train.shape[-1], output_dim=y_train.shape[-1], return_sequences=True)
    else:
        model = lstm_model(input_dim=x_train.shape[-1], output_dim=y_train.shape[-1], return_sequences=True)
    # Compile the model
    start = time.time()
    model.compile(loss='mean_squared_error', optimizer='adam')
    print('compilation time : ', time.time() - start)

    model.fit(
        x_train,
        y_train,
        epochs=3,
        validation_split=0.05)

    predictions = model.predict(x_test)

    helper.plot_lstm_prediction(y_test, predictions, chart_name=chart_name)

    trainScore = model.evaluate(x_train, y_train, verbose=0)
    # print('Train Score: %.8f MSE (%.8f RMSE)' % (trainScore, math.sqrt(trainScore)))
    pdf_output = io.StringIO()
    pdf_output.write('Train Score: %.8f MSE (%.8f RMSE)' % (trainScore, math.sqrt(trainScore)))

    testScore = model.evaluate(x_test, y_test, verbose=0)
    # print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))
    pdf_output.write('\nTest Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))
    range = [np.amin(emp_df['Close']), np.amax(emp_df['Close'])]

    # Calculate the stock price delta in $

    true_delta = testScore * (range[1] - range[0])
    # print('Delta Price: %.6f - RMSE * Adjusted Close Range' % true_delta)
    pdf_output.write('\nDelta Pr(ice: %.6f - RMSE * Adjusted Close Range' % true_delta)
    print("\n" + chart_name + ":\n" + pdf_output.getvalue())
    # output_map[chart_name]=print(pdf_output.getvalue())
    plot_model(model, to_file=chart_name + '_model.png', show_shapes=True, show_layer_names=True)


def train_and_predict(ticker):
    # ticker = 'AAPL'
    # 2021-01-04

    emp_df = ea.perform_empirical_analysis(ticker, '30m', 'polarities')
    # ea.plot_basic(emp_df,ticker+" Empirical Data")
    emp_df.to_csv(ticker + '_normalised.csv', index=False)
    emp_df = emp_df.drop(['Item'], axis=1)
    model_with_empirical_data(emp_df, False, 'Prediction using Empirical Data Only')
    model_with_empirical_data(emp_df, True, 'Tuned Prediction using Empirical Data Only')

    news = sa.calculatePolarities(sa.parseFinviz(ticker))
    # news=result.copy()

    news = news.resample("30min", on='datetime').mean().fillna(0)
    news.drop(news.between_time('16:00', '09:00').index, inplace=True)
    # news['Polarity']=np.where(news['compound']>0,1,np.where(news['compound']<0,-1,0))
    news = news.reset_index()
    polarities = news[['datetime', 'compound']]
    polarities.rename({'datetime': 'Datetime'}, axis=1, inplace=True)
    polarities.rename({'compound': 'Polarity'}, axis=1, inplace=True)

    emp_df = ea.perform_empirical_analysis(ticker, '30m', polarities)
    # ea.plot_basic(emp_df,ticker+" Emprical Data")
    emp_df.to_csv(ticker + '_normalised.csv', index=False)
    emp_df = emp_df.drop(['Item'], axis=1)
    model_with_empirical_data(emp_df, False, 'Prediction using Empirical Data and Sentiment Polarities')
    model_with_empirical_data(emp_df, True, 'Tuned Prediction using Empirical Data and Sentiment Polarities')
