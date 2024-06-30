import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix

# Description de l'application
st.title("Application de Classification avec Modèles d'Apprentissage Automatique")
st.write("""
Cette application permet de charger un fichier CSV, de prétraiter les données et de construire des modèles de classification.
Vous pouvez choisir entre la régression logistique et les arbres de décision, configurer les hyperparamètres, et visualiser les performances du modèle.
Développée par:
- Zakariae Jallal
- Youness Rajil
- Salah Eddine Jannani
- Younesse Ben Mohamed
""")

# Déclaration globale du modèle
model = None

# Chargement du fichier CSV
@st.cache
def load_data(file):
    return pd.read_csv(file)

# Fonction pour afficher la matrice de confusion sous forme de heatmap
def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Classe prédite')
    ax.set_ylabel('Classe réelle')
    ax.set_title('Matrice de confusion')
    st.pyplot(fig)

# Traitement des données avant l'entraînement du modèle
def preprocess_data(data):
    # Affichage des données et détection des valeurs manquantes
    st.subheader('Aperçu des données')
    st.write(data.head())

    st.subheader('Valeurs manquantes dans les données')
    st.write(data.isnull().sum())

    # Séparation des colonnes numériques et catégorielles
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

    # Supprimer une colonne nommée 'non' si elle existe
    if 'non' in data.columns:
        data = data.drop(columns=['non'], axis=1)
        st.write("La colonne 'non' a été supprimée.")

    # Remplacement des valeurs manquantes pour les colonnes numériques
    imputer_numeric = SimpleImputer(strategy='mean')
    data[numeric_columns] = imputer_numeric.fit_transform(data[numeric_columns])

    # Encodage des variables catégorielles
    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

    return data

# Interface utilisateur avec Streamlit
def main():
    global model

    st.title('Classification avec modèles ML')

    uploaded_file = st.file_uploader("Charger le fichier CSV", type=['csv'])

    if uploaded_file is not None:
        data = load_data(uploaded_file)

        # Prétraitement des données
        data = preprocess_data(data)

        columns = data.columns.tolist()

        # Choix des features et de la cible pour la prédiction
        st.subheader('Entraînement du modèle')
        features = st.multiselect('Choisir les features (X)', columns)
        target = st.selectbox('Choisir la colonne cible (y)', columns)

        if features and target:
            X = data[features]
            y = data[target]

            # Encoder la variable cible si nécessaire
            if y.dtype == 'object':
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
                classes = label_encoder.classes_
            else:
                classes = np.unique(y)

            # Sélectionner le modèle d'apprentissage
            model_type = st.selectbox('Choisir le modèle de classification', ['Régression Logistique', 'Arbre de Décision'])
            # Sélectionner le nombre de splits pour la validation croisée
            n_splits = st.slider('Nombre de splits pour la validation croisée', min_value=2, max_value=min(10, len(X)), value=5)
            if model_type == 'Régression Logistique':
                # Hyperparamètres de la Régression Logistique
                st.sidebar.subheader('Hyperparamètres de la Régression Logistique')
                c = st.sidebar.slider('C (Inverse de la régularisation)', 0.01, 10.0, 1.0)
                max_iter = st.sidebar.slider('Nombre maximal d\'itérations', 100, 1000, 100)

            elif model_type == 'Arbre de Décision':
                # Hyperparamètres de l'Arbre de Décision
                st.sidebar.subheader('Hyperparamètres de l\'Arbre de Décision')
                criterion = st.sidebar.selectbox('Critère de scission', ['gini', 'entropy'])
                splitter = st.sidebar.selectbox('Splitter', ['best', 'random'])
                max_depth = st.sidebar.slider('Profondeur maximale de l\'arbre', 1, 20, 5)
                min_samples_split = st.sidebar.slider('Nombre minimum d\'échantillons pour diviser un nœud', 2, 10, 2)
                min_samples_leaf = st.sidebar.slider('Nombre minimum d\'échantillons par feuille', 1, 10, 1)
                max_features = st.sidebar.selectbox('Max Features', [None, 'auto', 'sqrt', 'log2'])

            if st.button('Entraîner le modèle'):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if model_type == 'Régression Logistique':
                    model = LogisticRegression(C=c, max_iter=max_iter)
                elif model_type == 'Arbre de Décision':
                    model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                                   min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                   max_features=max_features)

                # Validation croisée StratifiedKFold
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                confusion = confusion_matrix(y_test, y_pred)

                st.subheader('Performance du modèle')
                st.write(f'Cross-Validation Accuracy: {cv_scores.mean():.2f}')
                st.write(f'Accuracy: {accuracy:.2f}')
                st.write(f'Precision: {precision:.2f}')
                st.write(f'F1 Score: {f1:.2f}')

                # Affichage de la matrice de confusion sous forme de heatmap
                plot_confusion_matrix(confusion, classes=classes)

if __name__ == '__main__':
    main()
