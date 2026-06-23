from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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
            ('minmax_scaler', MinMaxScaler(), ['age']),
            ('encoder', OneHotEncoder(), ['statut_marital', 'departement', 'poste', 'domaine_etude']),
        ],
        remainder='passthrough'
    )
    return preprocessor

def benchmark(pipeline, train_data, threshold = 0.5):
    scoring = {
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
    }

    cv_results = cross_validate(
        pipeline,
        train_data['X_train'],
        train_data['y_train'],
        cv=5,
        scoring=scoring,
    )

    print('CrossValidation Results : ')
    print(cv_results)
    print(f"Recall moyen : {cv_results['test_recall'].mean()}")
    print(f"F1-Score moyen : {cv_results['test_f1'].mean()}")

    pipeline.fit(train_data['X_train'], train_data['y_train'])

    y_pred = pipeline.predict(train_data['X_test'])

    # Check de la précision du modèle
    recall = recall_score(train_data['y_test'], y_pred)
    f1 = f1_score(train_data['y_test'], y_pred)

    print(f'Training Résults : ')
    print(f"Recall moyen : {recall}")
    print(f"F1-Score moyen : {f1}")
    print(confusion_matrix(train_data['y_test'], y_pred))

def train(pipeline, train_data, threshold = 0.5):

    pipeline.fit(train_data['X_train'], train_data['y_train'])

    y_pred = pipeline.predict(train_data['X_test'])

    # Check de la précision du modèle
    recall = recall_score(train_data['y_test'], y_pred)
    f1 = f1_score(train_data['y_test'], y_pred)

    print(f'Training Résults : ')
    print(f"Recall moyen : {recall}")
    print(f"F1-Score moyen : {f1}")

    print(confusion_matrix(train_data['y_test'], y_pred))

    # Prédictions sur le set d'entraînement via le pipeline
    y_pred_train = pipeline.predict(train_data['X_train'])

    # Prédictions sur le set de test via le pipeline
    y_pred_test = pipeline.predict(train_data['X_test'])
    print(f"Train F1-Score: {f1_score(train_data['y_train'], y_pred_train)}")
    print(f"Test F1-Score: {f1_score(train_data['y_test'], y_pred_test)}")

    # Prédiction des probabilités
    y_pred_proba = pipeline.predict_proba(train_data['X_test'])[:, 1]

    # Calcul de l'AUC pour vérifier la performance du modele : proche de 0.5 = proche d'un jeté de piece
    auc = roc_auc_score(train_data['y_test'], y_pred_proba)
    print(f"L'AUC du modèle est : {auc}")

# Affiche les features d'importance
def show_importances(features, names):
    df_importance = pd.DataFrame({
        'Variable': names,
        'Importance': features
    })

    df_importance = df_importance.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Importance',
        y='Variable',
        data=df_importance.head(10),
        hue='Variable',
        palette='viridis',
        legend=False
    )

    plt.title("Top 10 des variables les plus importantes")
    plt.xlabel("Importance")
    plt.ylabel("Variables")
    plt.tight_layout()
    plt.show()