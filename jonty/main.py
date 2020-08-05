"""
This code runs based on the assumption that the sentiments and reasons
calculated in the original dataset are reliable.
"""

import csv
import numpy as np
import pandas as pd
from collections import Counter
from itertools import repeat, chain
import re


def import_dataset() -> pd.DataFrame:
    tweetset = pd.read_csv('Tweets.csv', sep=',', quotechar='"')
    return tweetset


def remove_below_confidence(tweetset=pd.DataFrame(),
                      heading='airline_sentiment_confidence',
                      cutoff=0.9) -> pd.DataFrame:
    """
    Returns the dataset, removing any tweets with confidence values lower than
    or equal to the cutoff under the given heading.
    Does not remove tweets with NaN under the heading.
    """
    return tweetset.loc[(tweetset[heading] >= cutoff) |
                        (tweetset[heading].isnull())]


def remove_above_confidence(tweetset=pd.DataFrame(),
                      heading='airline_sentiment_confidence',
                      cutoff=0.9) -> pd.DataFrame:
    """
    Returns the dataset, removing any tweets with confidence values greater than the cutoff
    under the given heading. Does not remove tweets with NaN under the heading.
    """
    return tweetset.loc[(tweetset[heading] < cutoff) |
                        (tweetset[heading].isnull())]


def sentiment_proportion(tweetset=pd.DataFrame()) -> pd.DataFrame:
    """
    Presents each airline's sentiment levels as
    percentages of all tweets directed at the airline
    """
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


def destroy_retweets(tweetset=pd.DataFrame()) -> pd.DataFrame:
    """
    Removes all replies, that is, tweets containing unicode 'RT @'.
    """
    return tweetset.loc[tweetset['text'].str.contains('^(?!RT @).*$')]


def destroy_replies(tweetset=pd.DataFrame()) -> pd.DataFrame:
    """
    Removes all replies, that is, tweets containing unicode '“@*”'.
    """
    return tweetset.loc[tweetset['text'].str.contains('^(?!“@.*”).*$')]


def tweet_count(tweetset=pd.DataFrame) -> dict:
    """
    Lists the number of tweets directed at each airline in descending order
    """
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


def seq_ngrams(xs, n):
    """
    Taken from https://skeptric.com/ngram-python/
    """
    return [xs[i:i+n] for i in range(len(xs)-n+1)]


def shingle(text, w):
    tokens = text.split(' ')
    return [' '.join(xs) for xs in seq_ngrams(tokens, w)]


def list_popularity(items=[], count=10):
    """
    Returns a dictionary of the `count` most popular items in `l`,
    with their frequencies, in frequency-descending order
    """
    return {j: k for j, k in Counter(items).most_common(count)}


def find_ngrams(tweetset=pd.DataFrame(),
                ngram_count=5,
                n=10,
                heading='airline_sentiment_confidence',
                cutoff=0.4) -> dict:
    """
    Finds `ngram_count` `n`-grams in tweets with `heading` scores below `cutoff`.
    """
    tweetset_copy = destroy_replies(tweetset)
    tweetset_copy = destroy_retweets(tweetset_copy)
    print('The ' + str(ngram_count) + ' most frequently occurring ' +
          str(n) + '-grams in original tweets with ' + heading + ' scores below ' + str(cutoff) + ':')
    tweets = remove_above_confidence(tweetset_copy, heading=heading, cutoff=cutoff)['text'].to_numpy()

    ngrams = [ngram for text in tweets for ngram in shingle(
        text, n)]
    frequencies = list_popularity(ngrams, ngram_count)

    for j, k in frequencies.items():
        print('  ' + j + ' - ' + str(k))
    return ngrams


def main():
    tweetset = import_dataset()
    # Remove tweets with airline_sentiment_confidence below 0.75
    # tweetset = remove_above_confidence(tweetset, cutoff=0.75)
    sentiment_proportion(tweetset)
    print()
    tweet_count(tweetset)
    find_ngrams(tweetset, cutoff=1.0)


if __name__ == "__main__":
    main()
