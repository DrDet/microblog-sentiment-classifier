from sklearn.metrics import classification_report
import pandas as pd

from src.sentiment_classifier import SentimentClassifier


def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    return df[df['Sentiment'] != 'irrelevant']


def main(train_set_path: str, test_set_path: str, normalization_lexicon_path: str):
    train = load_dataset(train_set_path).sample(100)
    test = load_dataset(test_set_path)
    clf = SentimentClassifier(normalization_lexicon_path)
    clf.fit(train)
    pred_sent, pred_org = clf.predict_sentiment(test, True), clf.predict_organization(test)
    print('=========== Sentiment prediction report ===========')
    print(classification_report(test['Sentiment'], pred_sent))
    print('=========== Organization prediction report ===========')
    print(classification_report(test['Topic'], pred_org))


if __name__ == '__main__':
    main(train_set_path="data/Train.csv", test_set_path="data/Test.csv",
         normalization_lexicon_path="data/normalization-lexicon/emnlp_dict.txt")
