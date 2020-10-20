from src.sentiment_classifier import SentimentClassifier
import pandas as pd


def main(train_set_path: str, test_set_path: str, normalization_lexicon_path: str):
    train = pd.read_csv(train_set_path)
    test = pd.read_csv(test_set_path)
    clf = SentimentClassifier(normalization_lexicon_path)
    clf.fit(train, train['Sentiment'])
    print(clf.score(test, test['Sentiment']))


if __name__ == '__main__':
    main(train_set_path="data/Train.csv", test_set_path="data/Test.csv",
         normalization_lexicon_path="data/normalization-lexicon/emnlp_dict.txt")
