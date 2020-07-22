import csv
import numpy as np
import pandas as pd

tweetset = pd.read_csv('Tweets.csv', sep=',', quotechar='"')


def sentiment_proportion() -> pd.DataFrame:
    print('Tweet sentiment proportions for each airline:')
    airlines = set(tweetset['airline'])
    sentiment_percentages = pd.DataFrame(
        columns=['airline', 'positive', 'neutral', 'negative'])

    for airline in airlines:
        tweets = tweetset.loc[tweetset['airline'] == airline]
        sentiments = tweets['airline_sentiment']

        size = len(sentiments.index)
        pos = len(sentiments.loc[sentiments == 'positive'].index) / size
        neu = len(sentiments.loc[sentiments == 'neutral'].index) / size
        neg = len(sentiments.loc[sentiments == 'negative'].index) / size

        sentiment_percentages.append({'airline': airline,
                                      'positive': pos,
                                      'neutral': neu,
                                      'negative': neg}, ignore_index=True)
        print('  ' + airline + ' - ' +
              'Positive: ' + str(round(pos * 100, 2)) + '%' + ', ' +
              'Neutral: ' + str(round(neu * 100, 2)) + '%' + ', ' +
              'Negative: ' + str(round(neg * 100, 2)) + '%')

    return sentiment_percentages


def tweet_count() -> dict:
    print('Number of tweets directed at each airline:')
    airlines = set(tweetset['airline'])
    counts = {}
    for airline in airlines:
        tweet_count = len(tweetset.loc[tweetset['airline'] == airline].index)
        counts[airline] = tweet_count

    # Order counts
    counts = {k: v for k, v in sorted(
        counts.items(), key=lambda item: item[1], reverse=True)}
    for k, v in counts.items():
        print('  ' + k + ' - ' + str(v))
    return counts


def main():
    sentiment_proportion()
    print()
    tweet_count()


if __name__ == "__main__":
    main()
