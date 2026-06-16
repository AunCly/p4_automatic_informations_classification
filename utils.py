from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, recall_score, f1_score, precision_score, accuracy_score, classification_report
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, OneHotEncoder


# Fonction pour entrainer notre modèle et évaluer ses performances
def train_model(pipeline, d):
    # Entrainement du modèle
    pipeline.fit(d['X_train'], d['y_train'])

    # Prédictions sur le jeu de test
    y_pred = pipeline.predict(d['X_test'])

    # Check de la précision du modèle
    accurracy = accuracy_score(d['y_test'], y_pred)
    recall = recall_score(d['y_test'], y_pred)
    f1 = f1_score(d['y_test'], y_pred)

    print(f"Accuracy moyenne : {accurracy}")
    print(f"Recall moyen : {recall}")
    print(f"F1-Score moyen : {f1}")

def split_train_data(data, target) :
    labels = data[target]
    train_data = data.drop(target, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.2, random_state=42)

    data_train = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
    }

    return data_train

def create_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ('robust_scaler', RobustScaler(), ['revenu_mensuel']),
            ('minmax_scaler', MinMaxScaler(), ['age', 'distance_domicile_travail']),
            ('encoder', OneHotEncoder(), ['statut_marital', 'departement', 'poste', 'domaine_etude']),
        ],
        remainder='passthrough'
    )
    return preprocessor


# Compare les performances des différents modèles
def benchmark(models, preprocessor, train_data):
    for model in models:
        print(f"Benchmarking model: {model['name']}")

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model['model']),
        ])

        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0)
        }

        cv_results = cross_validate(
            pipeline,
            train_data['X_train'],
            train_data['y_train'],
            cv=5,
            scoring=scoring,
        )

        print(f"Accuracy moyenne : {cv_results['test_accuracy'].mean()}")
        print(f"Recall moyen : {cv_results['test_recall'].mean()}")
        print(f"F1-Score moyen : {cv_results['test_f1'].mean()}")
        print('------------------')