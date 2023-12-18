import pandas as pd
from sklearn import model_selection, ensemble, metrics


def train(df_path: str, seed: int) -> ensemble.RandomForestClassifier:
    """
    :param df_path: path to csv file to be used as a dataset
    :param seed: seed for random initialization
    :return: classifier
    """
    df = pd.read_csv(df_path)

    num_mult = len(df[df['label'] == 'multicolor'])
    num_mono = len(df[df['label'] == 'monochrome'])
    balanced = df.iloc[list(range(0, num_mult)) + list(range(num_mono, num_mono + num_mult))]

    x_train, x_test, y_train, y_test = model_selection.train_test_split(balanced.drop(['name', 'label'], axis=1),
                                                                        balanced['label'], test_size=0.2, shuffle=True,
                                                                        random_state=seed)

    clf = ensemble.RandomForestClassifier(random_state=seed)
    param_grid = {
        'n_estimators': [1, 5, 10, 20, 30, 50, 100],
        'max_depth': [1, 5, 10, 20, 30]
    }

    search = model_selection.GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
    search.fit(x_train, y_train)
    bp = search.best_params_

    clf = ensemble.RandomForestClassifier(max_depth=bp['max_depth'], n_estimators=bp['n_estimators'], random_state=seed)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.classification_report(y_test, y_pred))

    return clf
