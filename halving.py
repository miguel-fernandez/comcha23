import numpy as np

## NECESSARY: To understand HalvingRandomSearchCV is an experimental feature
from sklearn.experimental    import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.ensemble        import AdaBoostClassifier
from setup_data              import setup_digits_data


## nc = 10; n = 1797
X, y = setup_digits_data()

clf = AdaBoostClassifier(random_state=42)

## N = 460
param_distributions = {"learning_rate": np.arange(0.01, 0.51, 0.01),
                       "n_estimators" : range(20, 255, 5)}


search = HalvingRandomSearchCV(clf, param_distributions, verbose=1)
search.fit(X, y)

print(search.best_params_)
print(search.best_score_)
print(search.cv_results_)
