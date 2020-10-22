from sklearn.metrics import classification_report
import pandas as pd

from src.sentiment_classifier import SentimentClassifier


def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    return df[df['Sentiment'] != 'irrelevant']


def main(train_set_path: str, test_set_path: str, normalization_lexicon_path: str):
    train = load_dataset(train_set_path)
    clf = SentimentClassifier(normalization_lexicon_path)
    print('fitting classifier...')
    clf.fit(train)
    while True:
        print('What would you like to predict? (s - sentiment, o - organization)')
        mode = input()
        assert mode in ['s', 'o']
        print('Enter tweet text:')
        tweet_text = input()
        str_time, org = ('', '')
        if mode == 's':
            print('Enter time in format like "Wed Oct 19 16:56:52 +0000 2011":')
            str_time = input()
            print('Enter organization, must be one of the following: "apple" "google" "microsoft" "twitter":')
            org = input()
        X = pd.DataFrame({'TweetText': [tweet_text], 'Topic': [org], 'TweetDate': [str_time]})
        if mode == 's':
            pred_sent, rate = clf.predict_sentiment(X)[0], clf.predict_sentiment(X, get_rate=True)[0]
            print('%s: %.2f / 5.00' % (pred_sent, rate))
        else:
            print(clf.predict_organization(X)[0])
        print('Continue? (Y/n)')
        cont = input()
        if cont == 'n':
            break


if __name__ == '__main__':
    main(train_set_path="data/Train.csv", test_set_path="data/Test.csv",
         normalization_lexicon_path="data/normalization-lexicon/emnlp_dict.txt")
