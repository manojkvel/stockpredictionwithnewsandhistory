Stock Market is very volatile and is very difficult to analyze and predict its trends. Investors and market analysts study the market behavior and plan their investment strategies accordingly. As stock market produces large amount of data every day, it is very difficult for an individual to consider all the current and past information for predicting future trend of a stock. 
Stock Market Analysis can be broadly classified into 3 types - Fundamental Analysis, Technical Analysis and Quantitative Analysis. 
Fundamental Analysis involves analyzing the company’s financial data and market sentiments to get insights.
Technical Analysis involved mining of empirical data to get insights into the trends. 
Quantitative Analysis involves analyzing the volatility factors using stochastic methods. 

Stock Trend Analysis is an ongoing research field and has numerous challenges given volume, veracity and velocity of data involved. News has always been an important source of information to build perception of market investments. As the volumes of news and news sources are increasing rapidly, it's becoming impossible for an investor or even a group of investors to find out relevant news from the big chunk of news available. But, it's important to make a rational choice of investing timely in order to make maximum profit out of the investment plan. And having this limitation, computation comes into the place which automatically extracts news from all possible news sources, filter and aggregate relevant ones, analyze them in order to give them real time sentiments. 

This dissertation is an attempt to build an efficient model that scores emotions from news polarity which may affect changes in stock trends. In this project we follow the Fundamental analysis technique to discover future trend of a stock by considering news articles about a company as prime information and tries to classify news as positive , neutral and negative. If the news sentiment is positive, there are more chances that the stock price will go up and if the news sentiment is negative, then stock price may go down. We also implement technical analysis where applicable to capture any correlation between the news polarity, stock sentiment and stock price empirically. In this project we utilize Long-Short Term Memory (LSTM) Neural Network algorithm to predict stock prices - using just the empirical data and then with news sentiment polarities as well. 

Python Version : 3.6 + <br>
Frameworks used:
<ul><li>pandas</li>
<li>numpy</li>
<li>keras</li>
<li>nltk</li>
<li>matplotlib</li>
<li>sklearn</li>
<li>urllib</li>
<li>bs4</li>
<li>ssl</li>
<li>yfinance</li>
<li>flask</li></ul>



<h4><u>Sentiment Analysis using nltk Vader</u>:</h4>

Sentiment Analysis is the process of ‘computationally’ determining whether a piece of writing is positive, negative or neutral. In this project we use NLTK’s VADER library to perform sentiment analysis. VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in digital and social media. It is used for sentiment analysis of text which has both the polarities i.e. positive/negative. VADER is used to quantify how much of positive or negative emotion the text has and also the intensity of emotion. VADER works by the general applicability of the lexical features responsible for sentiments using a 'Wisdom of the Crowd' (WotC) approach. WotC relies on the idea that the collective knowledge of a group of people as expressed through their aggregated opinions can be trusted as an alternative to expert knowledge. This helps acquire a valid point estimate for the sentiment valence score of each context-free text. Amazon Mechanical Turk (MTurk) is one such famous crowdsourcing marketplace where distributed expert raters perform tasks like rating speeches remotely.

<h4><u>Deep Learning Models</u>:</h4>

LSTM RNN is chosen in this project instead of plain Recurrent Neural Nets (RNN) which are used specifically for sequence and pattern learning although they are networks with loops in them, allowing information to persist and thus ability to memorise the data accurately. Recurrent Neural Nets have vanishing Gradient descent problem which does not allow it to learn from past data as was expected. The remedy of this problem was solved in Long-Short Term Memory Networks, usually referred as LSTMs. These are a special kind of RNN, capable of learning long-term dependencies.
In addition to adjusting the architecture of the Neural Network, the following full set of
parameters can be tuned to optimize the prediction model:
<ul>
Number of Layers,
Number of Nodes,
Training Parameters,
Training / Test Split,
Validation Sets (kept constant at 0.05% of training sets)
,Batch Size,
Optimizer Function,
Epochs</ul>