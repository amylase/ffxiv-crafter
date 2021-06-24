import pickle

from scipy.stats import kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from ffxivcrafter.modeling.model import extract_feature, score_to_target, LinearEvaluator


def main():
    with open("../../data/samples.pkl", "rb") as f:
        samples = pickle.load(f)

    states, X, y = [], [], []
    for state, score in samples:
        states.append(state)
        X.append(extract_feature(state))
        y.append(score_to_target(score))
    state_train, state_test, X_train, X_test, y_train, y_test = train_test_split(states, X, y, test_size=0.2, random_state=42)

    model = LinearRegression(normalize=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(mean_absolute_error(y_test, y_pred))
    print(mean_squared_error(y_test, y_pred))
    print(kendalltau(y_test, y_pred))

    evaluator = LinearEvaluator(weights=model.coef_.tolist(), intercept=model.intercept_)
    with open("../../data/evaluator.pkl", "wb") as f:
        pickle.dump(evaluator, f)


if __name__ == '__main__':
    main()