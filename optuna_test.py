import optuna

from sklearn.ensemble        import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from setup_data              import setup_digits_data


## nc = 10; n = 1797
X, y = setup_digits_data()

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)


def objective(trial):
    clf = AdaBoostClassifier(random_state=42)

    lr   = trial.suggest_float("learning_rate", 0.3, 0.8)
    nest = trial.suggest_int("n_estimators", 20, 250)

    clf = AdaBoostClassifier(learning_rate=lr, n_estimators=nest, random_state=42)
    clf.fit(Xtrain, ytrain)

    test_score = clf.score(Xtest, ytest)
    
    return test_score


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print(study.best_trial.value)
    print(study.best_trial.params)

    #import optuna.visualization as optuna_viz

    #fig = optuna_viz.plot_slice(study)
    #fig.show()
