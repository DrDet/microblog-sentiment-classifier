# What is it?
It's a sentiment classifier of tweets.

It can predict:
- sentiment of tweet (negative | neutral | positive)
and assign grade to it (from 1 to 5). 
- topic of tweet (apple | google | microsoft | twitter).
# Setup
Pipenv virtual environment is used here.
- install pipenv
```
pip install pipenv
```
- setup virtual env
```
pipenv install
pipenv shell
```
It's time to run it! 
# Run
It has a very simple command line interface:
```
python3 main.py
```
Also it can be executed in jupiter notebook with some reports and samples:
```
jupyter notebook main.ipynb
```