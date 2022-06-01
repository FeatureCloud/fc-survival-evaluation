# Survival Prediction Evaluation FeatureCloud App

[![unstable](http://badges.github.io/stability-badges/dist/unstable.svg)](http://github.com/badges/stability-badges)

## Description
An Evaluation FeatureCloud app for survival/time-to-event predictions. 

This allows evaluating your trained models using the following metrics:

- c-index (concordance index)
    - local c-index
    - local concordant pairs
    - global c-index equally weighted (mean)
    - global c-index weighted by the number of samples at each client
    - global c-index weighted by the number of concordant pairs at each client

## Input
- test.csv containing the actual test dataset with an event and time column
- pred.csv containing the predictions of the model on the test dataset

## Output
- scores.tsv containing various evaluation metrics

## Workflows
Can be combined with the following apps:
- Pre: Survival SVM

This app is compatible with CV.

## Config
Use the config file to customize the evaluation. Just upload it together with your training data as `config.yml`
```yaml
fc_survival_evaluation:
  privacy:
    min_concordant_pairs: 3  # minimum: 3; threshold of concordant pairs for participation
  input:
    y_test: "test.csv"
    y_pred: "pred.csv"  # could be the same as y_test if predictions were appended to test data
  format:
    sep: ","
    label_survival_time: "tte"
    label_event: "event"
    event_value: 'True'  # optional, default='True'; value of an entry in the event column when an event occurred
    event_censored_value: 'False'  # optional, default='False'; value of an entry in the event column when censored
    label_predicted_time: "predicted_tte"
  parameters:
    objective: regression  # can be regression or ranking
  split:
    mode: directory  # directory if cross validation was used before, else file
    dir: data  # data if cross validation app was used before, else .
```
