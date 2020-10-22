import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect
from datetime import datetime

from .tweet_cleaner import TweetCleaner


class SentimentClassifier:
    LANGS = {
        'hu': 'hungarian',
        'sv': 'swedish',
        'kk': 'kazakh',
        'fi': 'finnish',
        'no': 'norwegian',
        'ar': 'arabic',
        'id': 'indonesian',
        'pt': 'portuguese',
        'tr': 'turkish',
        'az': 'azerbaijani',
        'sl': 'slovene',
        'es': 'spanish',
        'da': 'danish',
        'ne': 'nepali',
        'ro': 'romanian',
        'el': 'greek',
        'nl': 'dutch',
        'tg': 'tajik',
        'de': 'german',
        'en': 'english',
        'ru': 'russian',
        'fr': 'french',
        'it': 'italian',
    }

    ORGS = {
        'apple': 0.1,
        'google': 0.2,
        'microsoft': 0.3,
        'twitter': 0.4
    }

    def __init__(self, normalization_lexicon_path: str, ngram_range=(1, 1), sentiment_rate_range=(1,5), start_day_hour=8):
        self._start_day_hour = start_day_hour
        self._sentiment_rate_range = sentiment_rate_range
        self._clf_sentiment = SVC(gamma="scale", probability=True)
        self._clf_organization = SVC(gamma="scale")
        self._cleaner = TweetCleaner()
        self._lexicon = {}
        self._tknzr = TweetTokenizer()
        self._vectorizer = TfidfVectorizer(ngram_range=ngram_range, sublinear_tf=True)
        with open(normalization_lexicon_path, "r", encoding="utf-8") as lexicon_file:
            for line in lexicon_file:
                informal, formal = line.split("\t")
                self._lexicon[informal] = formal

    def fit(self, tweets_dataset):
        corpus = []
        sentiment_labels = []
        organization_features = []
        temporal_features = []
        organization_labels = []
        for idx, tweet in tweets_dataset.iterrows():
            preprocessed_tweet = self._preprocess(tweet['TweetText'], discard_short_tweets=True)
            if preprocessed_tweet is not None:
                corpus.append(preprocessed_tweet)
                sentiment_labels.append(tweets_dataset['Sentiment'][idx])
                org = tweets_dataset['Topic'][idx]
                organization_labels.append(org)
                organization_features.append(self.ORGS[org])
                temporal_features.append(self._calc_temporal_feature(tweets_dataset['TweetDate'][idx]))
        features = self._vectorizer.fit_transform(corpus)
        features_sentiment = pd.DataFrame(features.todense()).join(pd.DataFrame({'Topic': organization_features}))
        features_sentiment = features_sentiment.join(pd.DataFrame({'TweetDate': temporal_features}))
        self._clf_sentiment.fit(features_sentiment, sentiment_labels)
        self._clf_organization.fit(features, organization_labels)

    def predict_sentiment(self, X, get_rate=False):
        features = self._prepare_for_prediction(X)
        features = pd.DataFrame(features.todense()).join(X['Topic'].apply(lambda x: self.ORGS[x]).reset_index(drop=True))
        features = features.join(X['TweetDate'].apply(self._calc_temporal_feature).reset_index(drop=True))
        if get_rate:
            probs = self._clf_sentiment.predict_proba(features)
            return list(map(self._calc_sentiment_rate_from_probs, probs))
        return self._clf_sentiment.predict(features)

    def predict_organization(self, X):
        features = self._prepare_for_prediction(X)
        return self._clf_organization.predict(features)

    def _calc_sentiment_rate_from_probs(self, probs):
        p_negative = probs[np.where(self._clf_sentiment.classes_ == 'negative')[0][0]]
        p_neutral = probs[np.where(self._clf_sentiment.classes_ == 'neutral')[0][0]]
        p_positive = probs[np.where(self._clf_sentiment.classes_ == 'positive')[0][0]]
        min_rate, max_rate = self._sentiment_rate_range
        middle_rate = (min_rate + max_rate) / 2.0
        return p_negative * min_rate + p_neutral * middle_rate + p_positive * max_rate

    def _calc_temporal_feature(self, str_time):
        t = datetime.strptime(str_time, '%a %b %d %H:%M:%S %z %Y')
        feature = (t.hour - self._start_day_hour) % 24 + 1  # from 1 to 24
        return feature / (24 * 2.0)                         # from 0.02 to 0.5 (normalized)

    def _prepare_for_prediction(self, X):
        corpus = [self._preprocess(tweet['TweetText']) for idx, tweet in X.iterrows()]
        features = self._vectorizer.transform(corpus)
        return features

    def _preprocess(self, text: str, discard_short_tweets=False):
        text = self._cleaner.clean_up(text)

        word_tokens = self._tknzr.tokenize(text)

        try:
            lang = detect(text)
        except:
            lang = 'en'
        stop_words = set(stopwords.words(self.LANGS.get(lang, 'english')))
        word_tokens = [word for word in word_tokens if word not in stop_words]

        word_tokens = [self._lexicon[word] if word in self._lexicon else word for word in word_tokens]

        if discard_short_tweets and len(word_tokens) < 3:
            return None

        return " ".join(word_tokens)
