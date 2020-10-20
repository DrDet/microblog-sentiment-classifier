from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

from .tweet_cleaner import TweetCleaner


class SentimentClassifier:
    def __init__(self, normalization_lexicon_path: str, ngram_range=(1, 1)):
        self._clf = SVC(gamma="scale")
        self._cleaner = TweetCleaner()
        self._lexicon = {}
        self._tknzr = TweetTokenizer()
        self._vectorizer = TfidfVectorizer(ngram_range=ngram_range, sublinear_tf=True)
        with open(normalization_lexicon_path, "r", encoding="utf-8") as lexicon_file:
            for line in lexicon_file:
                informal, formal = line.split("\t")
                self._lexicon[informal] = formal

    def fit(self, X, y):
        corpus = []
        labels = []
        for idx, tweet in X.iterrows():
            preprocessed_tweet = self._preprocess(tweet['TweetText'], discard_short_tweets=True)
            if preprocessed_tweet is not None:
                corpus.append(preprocessed_tweet)
                labels.append(y[idx])
        features = self._vectorizer.fit_transform(corpus)
        self._clf.fit(features, labels)

    def predict(self, X):
        features = self._prepare_for_prediction(X)
        return self._clf.predict(features)

    def score(self, X, y):
        features = self._prepare_for_prediction(X)
        return self._clf.score(features, y)

    def _prepare_for_prediction(self, X):
        corpus = [self._preprocess(tweet['TweetText']) for idx, tweet in X.iterrows()]
        features = self._vectorizer.transform(corpus)
        return features

    def _preprocess(self, text: str, discard_short_tweets=False):
        text = self._cleaner.clean_up(text)

        word_tokens = self._tknzr.tokenize(text)

        stop_words = set(stopwords.words('english'))
        word_tokens = [word for word in word_tokens if word not in stop_words]

        word_tokens = [self._lexicon[word] if word in self._lexicon else word for word in word_tokens]

        if discard_short_tweets and len(word_tokens) < 3:
            return None

        return " ".join(word_tokens)
