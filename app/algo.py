import numpy as np
import pandas as pd
import plotly.graph_objects as go


def check(y_test, y_proba):
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.to_numpy().ravel()
    if isinstance(y_proba, pd.DataFrame):
        y_proba = y_proba.to_numpy().ravel()

    try:
        if y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]
    except IndexError:
        pass

    return y_test, y_proba


def compute_local_prediction_error(actual, pred):
    pred_errors = []
    for i in range(len(actual)):
        pred_errors.append(pred[i] - actual[i])

    return pred_errors


def aggregate_prediction_errors(local_results):
    return np.concatenate(local_results)


def mae(pred_errors):
    sum_error = 0.0
    for i in range(len(pred_errors)):
        sum_error += abs(pred_errors[i])
    return sum_error / float(len(pred_errors))


def max_error(pred_errors):
    absolut = [abs(ele) for ele in pred_errors]
    return np.max(absolut)


def rmse(pred_errors):
    sum_error = 0.0
    for i in range(len(pred_errors)):
        sum_error += (pred_errors[i] ** 2)
    mean_error = sum_error / float(len(pred_errors))
    return np.sqrt(mean_error)


def mse(pred_errors):
    sum_error = 0.0
    for i in range(len(pred_errors)):
        sum_error += (pred_errors[i] ** 2)
    mean_error = sum_error / float(len(pred_errors))
    return mean_error


def medae(pred_errors):
    absolut = [abs(ele) for ele in pred_errors]
    return np.median(absolut)


def create_score_df(pred_errors):
    mae_score = mae(pred_errors)
    max_score = max_error(pred_errors)
    rmse_score = rmse(pred_errors)
    mse_score = mse(pred_errors)
    medae_score = medae(pred_errors)

    scores = ["mean_absolut_error", "max_error", "root_mean_squared_error", "mean_squared_error",
              "median_absolut_error"]
    data = [mae_score, max_score, rmse_score, mse_score, medae_score]

    df = pd.DataFrame(list(zip(scores, data)), columns=["metric", "score"])

    return df, data


def create_cv_accumulation(maes, maxs, rmses, mses, medaes):
    scores = [maes, maxs, rmses, mses, medaes]
    cols = ["mean_absolut_error", "max_error", "root_mean_squared_error", "mean_squared_error",
            "median_absolut_error"]

    df = pd.DataFrame(data=scores).transpose()
    df.columns = cols

    return df

def plot_boxplots(df, title):
    fig = go.Figure()
    fig.add_trace(go.Box(y=df["mean_absolut_error"].map(np.log2), quartilemethod="linear", name="Mean Absolute Error"))
    fig.add_trace(go.Box(y=df["max_error"].map(np.log2), quartilemethod="linear", name="Max Error"))
    fig.add_trace(go.Box(y=df["root_mean_squared_error"].map(np.log2), quartilemethod="linear", name="Root Mean Squared Error"))
    #fig.add_trace(go.Box(y=df["mean_squared_error"].map(np.log2), quartilemethod="linear", name="f1-score"))
    fig.add_trace(go.Box(y=df["median_absolut_error"].map(np.log2), quartilemethod="linear", name="Median Absolut Error"))
    fig.update_layout(title=title, yaxis_title='Log2')

    return fig
