import numpy as np

## NECESSARY: To understand HalvingRandomSearchCV is an experimental feature
from sklearn.experimental    import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV


from sklearn.tree     import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from setup_data       import setup_digits_data


## nc = 10; n = 1797
X, y = setup_digits_data()

pipe = Pipeline([("classifier", AdaBoostClassifier())])

space1 = {"classifier": [AdaBoostClassifier(DecisionTreeClassifier())],
          "classifier__estimator__max_features": range(1,3),
          "classifier__estimator__max_depth": range(1,11),
          "classifier__learning_rate": np.arange(0.01, 0.51, 0.01),
          "classifier__n_estimators" : range(20, 255, 5),
          "classifier__random_state" : [42]}

space2 = {"classifier": [GradientBoostingClassifier()],
          "classifier__n_estimators" : [int(round(x,0)) \
                                        for x in np.logspace(2,4,num=15)],
          "classifier__max_features" : range(1,3),
          "classifier__max_depth"    : range(1,11),
          "classifier__learning_rate": np.arange(.1,.91,0.1)}


search_results = []
for space in [space1, space2]:
    search = HalvingRandomSearchCV(pipe, space, verbose=1)
    search.fit(X, y)

    search_results += [{"params": search.best_params_,
                        "score" : search.best_score_}]

print(search_results)
