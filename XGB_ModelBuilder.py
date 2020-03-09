def XGB_ModelBuilder(X_train, y_train, X_test, y_test, X_unknown=[]):

    # XGB_ModelBuilder.py
    # Created by KAC on 02/12/2020

    """ This function takes in data and completes a grid search to tune parameters automatically. It then makes predictions
    and calculates an MAE score for those predictions."""

    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import RFECV
    from sklearn.metrics import log_loss
    from xgboost import XGBClassifier as XGB
    from sklearn.model_selection import cross_val_score, RandomizedSearchCV
    from sklearn.metrics import make_scorer

    # scorer = make_scorer(log_loss, greater_is_better=False)
    XGB_model = XGB()
    selector = RFECV(estimator=XGB_model, scoring='neg_log_loss', cv=5)
    selector.fit(X_train, y_train)
    CV_score = cross_val_score(selector, X_train, y_train, scoring='neg_log_loss', cv=5)
    scr = np.mean(CV_score)
    print(pd.DataFrame({'Variable':X_train.columns,
                  'Importance':selector.ranking_}).sort_values('Importance', ascending=True).head(50))
    print("Optimal number of features: ", selector.n_features_)
    print("Log Loss for All Features: ", scr)

    if selector.n_features_ < len(X_train.columns):
        X_train_transformed = selector.transform(X_train)
        X_test_transformed = selector.transform(X_test)

        CV_score = cross_val_score(selector, X_train_transformed, y_train, scoring='neg_log_loss', cv=5)
        scr = np.mean(CV_score)
        print("Log Loss for Selected Features on Training Data: ", scr)
    else:
        X_train_transformed = X_train
        X_test_transformed = X_test
        print("Not optimal to remove features. Proceeding to parameter tuning.")

    # Current Best: {'subsample': 0.9, 'n_estimators': 250, 'min_child_weight': 2, 'max_depth': 8, 'learning_rate': 0.02, 'colsample_bytree': 0.85}
    parameters = {"learning_rate": [0.01, 0.015, 0.02, 0.025, 0.03], #[0.01, 0.05, 0.1],
                  "n_estimators": [250, 500, 600], #[500, 750, 1000],
                  "max_depth": [8, 9, 10, 12], #[3, 6, 9],
                  "min_child_weight": [2, 5, 8], #[1, 2],
                  "colsample_bytree": [0.7, 0.75, 0.8, 0.85], #[0.5, 0.75, 1],
                  "subsample": [0.9, 1] #[0.5, 0., 1]
                  }
    rsearch = RandomizedSearchCV(estimator=XGB_model, param_distributions=parameters, scoring='neg_log_loss', n_iter=250, cv=5) #XGB_model
    rsearch.fit(X_train_transformed, y_train)
    print(rsearch.best_params_)

    CV_score = cross_val_score(rsearch, X_train_transformed, y_train, scoring='neg_log_loss', cv=5)
    scr = np.mean(CV_score)
    print("Log Loss for Selected Features and Parameter Tuning on Training Data: ", scr)

    predictions = rsearch.predict_proba(X_test_transformed)

    pred_scr = round(log_loss(y_test, predictions), 5)
    print("2019 Score: ", pred_scr)

    if X_unknown is not None:
        X_final = pd.concat([X_train, X_test])
        X_final = RFECV.transform(X_final)
        y_final = pd.concat([y_train, y_test])

        X_unknown = RFECV.transform(X_unknown)

        rsearch.fit(X_final, y_final)
        predictions_final = rsearch.predict(X_unknown)

    else:
        predictions_final = []

    return predictions, predictions_final
