import nltk
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import ssl

# NLTK VADER for sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

finviz_url = 'https://finviz.com/quote.ashx?t='


# Parse new_table from the finviz website
def parseFinviz(tickers):
    """

    :param tickers: list of stock tickers to fetch from finviz
    :return:
    """
    news_tables = {}
    tickers = ['AAPL']

    for ticker in tickers:
        url = finviz_url + ticker
        req = Request(url=url, headers={'user-agent': 'my-app/0.0.1'})
        gcontext = ssl.SSLContext()  # Only for gangstars
        response = urlopen(req, context=gcontext)
        # Read the contents of the file into 'html'
        html = BeautifulSoup(response)
        # Find 'news-table' in the Soup and load it into 'news_table'
        news_table = html.find(id='news-table')
        # Add the table to our dictionary
        news_tables[ticker] = news_table

    # Read one single day of headlines for given ticker
    arg = news_tables['AAPL']
    arg_tr = arg.findAll('tr')

    for i, table_row in enumerate(arg_tr):
        a_text = table_row.a.text
        td_text = table_row.td.text
        print(a_text.encode('utf-8'))
        print(td_text.encode('utf-8'))
        # Exit after printing 4 rows of data
        if i == 3:
            break

    parsed_news = []

    # Iterate through the news
    for file_name, news_table in news_tables.items():
        # Parse for 'news_table'
        for x in news_table.findAll('tr'):
            text = x.a.get_text()
            date_scrape = x.td.text.split()

            if len(date_scrape) == 1:
                time = date_scrape[0]

            else:
                date = date_scrape[0]
                time = date_scrape[1]
            ticker = file_name.split('_')[0]
            datetime = pd.to_datetime(date + ' ' + time)
            parsed_news.append([ticker, datetime, date, time, text])

    return parsed_news


def calculatePolarities(parsed_news):
    """

    :param parsed_news: news extracted from finviz for ticker in context
    :return:
    """
    # Instantiate the sentiment intensity analyzer, Convert the parsed_news list into a DataFrame called
    # 'parsed_and_scored_news' Iterate through the headlines and get the polarity scores using vader
    vader = SentimentIntensityAnalyzer()
    # columns = ['ticker', 'datetime','date', 'time', 'headline']
    columns = ['ticker', 'datetime', 'date', 'time', 'headline']
    parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)
    scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')
    # parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date
    # parsed_and_scored_news['time'] = pd.to_datetime(parsed_and_scored_news.time).dt.time
    with open(parsed_news[0][0] + '_news_headlines.csv', 'a') as f:
        f.write('\n')
        f.close()
    parsed_and_scored_news.to_csv(parsed_news[0][0] + '_news_headlines.csv', index=False, header=False, mode='a')
    result = pd.read_csv(parsed_news[0][0] + '_news_headlines.csv')
    result.datetime = pd.to_datetime(result.datetime)
    return result


def plotPolarities(parsed_and_scored_news):
    """

    :param parsed_and_scored_news: VADER scored headline polarities to plot against dates
    """
    plt.rcParams['figure.figsize'] = [6, 6]

    # Group by date and ticker columns from scored_news and calculate the mean
    mean_scores = parsed_and_scored_news.groupby(['ticker', 'date']).mean()

    mean_scores.to_csv('polarities.csv')
    # Unstack the column ticker
    mean_scores = mean_scores.unstack()

    # Get the cross-section of compound in the 'columns' axis
    mean_scores = mean_scores.xs('compound', axis="columns").transpose()

    # Plot a bar chart with pandas
    mean_scores.plot(kind='bar')
    plt.grid()
