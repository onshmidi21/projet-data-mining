import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import ast

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare les données : encodage, gestion des Unknown, etc."""
    df_clean = df.copy()
    
    # Convertir la colonne 'damage' en liste si c'est une chaîne
    def parse_damage(damage_str):
        if pd.isna(damage_str):
            return ['No damage']
        if isinstance(damage_str, list):
            return damage_str
        try:
            # Essayer d'évaluer comme une liste Python
            return ast.literal_eval(damage_str)
        except:
            # Si échec, retourner comme liste d'un élément
            return [str(damage_str)]
    
    df_clean['damage'] = df_clean['damage'].apply(parse_damage)
    
    # Déterminer le type de dommage
    df_clean['damage_type'] = df_clean['damage'].apply(
        lambda x: 'No damage' if x == ['No damage'] or 'No damage' in x else 'Damaged'
    )
    
    # Extraire l'âge numérique avec gestion des formats complexes d'OCR
    def extract_age(age_str):
        if pd.isna(age_str) or age_str in ['Unknown', 'Inconnu', 'Indéterminé']:
            return 0
        
        # Convertir en string pour traitement
        age_str = str(age_str)
        
        # Mapper les catégories d'âge vers des valeurs numériques approximatives
        age_mapping = {
            'Très ancien': 50,
            'Ancien': 20,
            'Classique': 25,
            'Moderne': 10,
            'Contemporain': 5,
            'Récent': 3,
            'Standard': 8
        }
        
        # Chercher les catégories d'âge dans la chaîne
        for category, value in age_mapping.items():
            if category.lower() in age_str.lower():
                # Si une année est mentionnée, calculer l'âge réel
                year_match = re.search(r'(\d{4})', age_str)
                if year_match:
                    year = int(year_match.group(1))
                    current_year = 2024
                    calculated_age = current_year - year
                    return max(1, calculated_age)
                return value
        
        # Chercher un nombre direct dans la chaîne
        number_match = re.search(r'(\d+)', age_str)
        if number_match:
            return int(number_match.group(1))
        
        return 0
    
    df_clean['age_numeric'] = df_clean['age'].apply(extract_age)
    
    # Remplacer les valeurs Unknown/Inconnu par des valeurs par défaut
    df_clean['brand'] = df_clean['brand'].replace(['Unknown', 'Inconnu'], 'Unknown')
    df_clean['country'] = df_clean['country'].replace(['Unknown', 'Inconnu', 'International'], 'Unknown')
    
    # Supprimer les lignes où brand ET country sont Unknown
    df_clean = df_clean[~((df_clean['brand'] == 'Unknown') & (df_clean['country'] == 'Unknown'))]
    
    return df_clean

def perform_clustering(df: pd.DataFrame) -> dict:
    """Clustering KMeans sur marque, âge et type de dommage."""
    df_clean = prepare_data(df)
    if len(df_clean) < 3:
        return {"error": "Données insuffisantes pour le clustering (minimum 3 échantillons)"}
    
    # Features pour clustering
    features = pd.get_dummies(df_clean[['brand', 'damage_type']])
    features['age'] = df_clean['age_numeric']
    
    # Normaliser les features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Déterminer le nombre optimal de clusters
    n_clusters = min(4, max(2, len(df_clean) // 20))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    
    df_clean['cluster'] = clusters
    
    # Résumé des clusters avec plus d'informations - CONVERTIR EN TYPES PYTHON NATIFS
    cluster_summary = {}
    for cluster_id in sorted(df_clean['cluster'].unique()):
        cluster_data = df_clean[df_clean['cluster'] == cluster_id]
        cluster_summary[str(cluster_id)] = {
            'brand': int(len(cluster_data)),  # Convertir en int natif
            'age_numeric': float(round(cluster_data['age_numeric'].mean(), 2)),  # float natif
            'damage_type': int((cluster_data['damage_type'] == 'Damaged').sum()),
            'top_brand': str(cluster_data['brand'].mode()[0]) if len(cluster_data) > 0 else 'Unknown',
            'top_country': str(cluster_data['country'].mode()[0]) if len(cluster_data) > 0 else 'Unknown'
        }
    
    return {
        "num_clusters": int(len(set(clusters))),
        "summary": cluster_summary,
        "interpretation": "Les clusters représentent des groupes de voitures similaires par marque, âge et état"
    }

def find_association_rules(df: pd.DataFrame) -> list:
    """Règles d'association Apriori sur marque + âge + dommages."""
    df_clean = prepare_data(df)
    if len(df_clean) < 5:
        return [{"warning": "Données insuffisantes pour les règles d'association (minimum 5 échantillons)"}]
    
    # Créer des bins d'âge
    df_clean['age_bin'] = pd.cut(
        df_clean['age_numeric'], 
        bins=[0, 5, 10, np.inf], 
        labels=['Jeune (0-5 ans)', 'Moyen (6-10 ans)', 'Ancien (11+ ans)'],
        include_lowest=True
    )
    
    # Supprimer les NaN
    df_clean = df_clean.dropna(subset=['age_bin'])
    
    if len(df_clean) < 5:
        return [{"warning": "Pas assez de données après nettoyage"}]
    
    # Créer des transactions
    transactions = []
    for _, row in df_clean.iterrows():
        transaction = [
            row['brand'],
            str(row['age_bin']),
            row['damage_type']
        ]
        transactions.append(transaction)
    
    # Encoder les transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Apriori avec support plus bas
    try:
        freq_items = apriori(df_trans, min_support=0.05, use_colnames=True)
        if freq_items.empty:
            return [{"warning": "Aucun itemset fréquent trouvé (essayez un support plus bas)"}]
        
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.4, num_itemsets=len(freq_items))
        
        if rules.empty:
            return [{"warning": "Aucune règle d'association trouvée"}]
        
        # Trier par lift décroissant
        rules = rules.sort_values('lift', ascending=False)
        
        # Convertir en format JSON-safe
        results = []
        for _, rule in rules.head(10).iterrows():
            antecedent_list = [str(item) for item in rule['antecedents']]
            consequent_list = [str(item) for item in rule['consequents']]
            
            results.append({
                "antecedent": antecedent_list,
                "consequent": consequent_list,
                "support": float(rule['support']),
                "confidence": float(rule['confidence']),
                "lift": float(rule['lift']),
                "interpretation": f"Si {', '.join(antecedent_list)}, alors {', '.join(consequent_list)} avec {round(rule['confidence']*100, 1)}% de confiance"
            })
        
        return results
    except Exception as e:
        return [{"warning": f"Erreur lors du calcul des règles: {str(e)}"}]

def predict_damage_risk(df: pd.DataFrame) -> dict:
    """Random Forest pour prédire risque de dommages."""
    df_clean = prepare_data(df)
    if len(df_clean) < 10:
        return {"error": "Données insuffisantes pour le modèle de prédiction (minimum 10 échantillons)"}
    
    # Target : 1 si damaged, 0 sinon
    df_clean['target'] = (df_clean['damage_type'] == 'Damaged').astype(int)
    
    # Vérifier si seulement une classe unique
    unique_classes = df_clean['target'].nunique()
    if unique_classes == 1:
        risk_class = int(df_clean['target'].iloc[0])
        avg_risk = float(risk_class)
        high_risk_cars = []
        if risk_class == 1:
            for _, row in df_clean.head(3).iterrows():
                high_risk_cars.append({
                    'image': str(row['image']),
                    'risk_score': 1.0
                })
        return {
            "warning": f"Une seule classe détectée ({'Endommagé' if risk_class == 1 else 'Sans dommage'}), risque uniforme.",
            "avg_risk": avg_risk,
            "high_risk_cars": high_risk_cars,
            "feature_importance": {}
        }
    
    # Encoder les features catégorielles
    le_brand = LabelEncoder()
    le_country = LabelEncoder()
    
    df_clean['brand_encoded'] = le_brand.fit_transform(df_clean['brand'])
    df_clean['country_encoded'] = le_country.fit_transform(df_clean['country'])
    
    X = df_clean[['brand_encoded', 'country_encoded', 'age_numeric']]
    y = df_clean['target']
    
    if len(X) < 2:
        return {"error": "Trop peu d'échantillons pour diviser en train/test"}
    
    # Split train/test
    test_size = min(0.3, max(0.1, 3 / len(X)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)
    
    # Entraîner le modèle
    rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    rf.fit(X_train, y_train)
    
    accuracy = float(rf.score(X_test, y_test))
    
    # Noms lisibles pour les features - CONVERTIR EN FLOAT
    feature_names = ['Marque', 'Pays', 'Âge']
    feature_importance = {name: float(importance) for name, importance in zip(feature_names, rf.feature_importances_)}
    
    # Prédictions sur tout le dataset
    proba = rf.predict_proba(X)
    if proba.shape[1] == 2:
        risks = proba[:, 1]
    else:
        risks = np.full(len(X), 0.5)
    
    df_clean['risk_score'] = risks
    
    # Top véhicules à risque - CONVERTIR EN TYPES NATIFS
    high_risk = df_clean.nlargest(5, 'risk_score')[['image', 'risk_score', 'brand', 'age_numeric']]
    high_risk_cars = []
    for _, row in high_risk.iterrows():
        high_risk_cars.append({
            'image': str(row['image']),
            'risk_score': float(row['risk_score']),
            'brand': str(row['brand']),
            'age_numeric': float(row['age_numeric'])
        })
    
    return {
        "accuracy": float(round(accuracy, 3)),
        "feature_importance": feature_importance,
        "avg_risk": float(round(risks.mean(), 3)),
        "high_risk_cars": high_risk_cars
    }

def detect_anomalies(df: pd.DataFrame) -> list:
    """Isolation Forest pour anomalies (ex: âges/plaques suspects)."""
    df_clean = prepare_data(df)
    if len(df_clean) < 5:
        return [{"warning": "Données insuffisantes pour la détection d'anomalies (minimum 5 échantillons)"}]
    
    # Features pour anomalies : plaque longueur, âge, etc.
    df_clean['plate_length'] = df_clean['plate'].astype(str).str.len()
    
    # Compter le nombre de dommages
    df_clean['damage_count'] = df_clean['damage'].apply(lambda x: len(x) if x != ['No damage'] else 0)
    
    features = df_clean[['age_numeric', 'plate_length', 'damage_count']].fillna(0)
    
    # Ajuster la contamination en fonction de la taille du dataset
    contamination = min(0.1, max(0.05, 5 / len(df_clean)))
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomalies_pred = iso_forest.fit_predict(features)
    
    # Calculer les scores d'anomalie
    anomaly_scores = iso_forest.decision_function(features)
    df_clean['anomaly_score'] = -anomaly_scores
    
    # Filtrer les anomalies
    anomalies_df = df_clean[anomalies_pred == -1].copy()
    
    if anomalies_df.empty:
        return [{"message": "Aucune anomalie détectée - Toutes les données sont cohérentes"}]
    
    # Trier par score d'anomalie décroissant
    anomalies_df = anomalies_df.sort_values('anomaly_score', ascending=False)
    
    # CONVERTIR EN TYPES NATIFS POUR PYDANTIC
    results = []
    for _, row in anomalies_df.head(10).iterrows():
        reason_parts = []
        
        if row['age_numeric'] > 15 or row['age_numeric'] == 0:
            reason_parts.append(f"Âge inhabituel ({int(row['age_numeric'])} ans)")
        
        if row['plate_length'] < 5 or row['plate_length'] > 15:
            reason_parts.append(f"Plaque anormale (longueur: {int(row['plate_length'])})")
        
        if row['damage_count'] > 3:
            reason_parts.append(f"Nombreux dommages ({int(row['damage_count'])})")
        
        if not reason_parts:
            reason_parts.append("Combinaison inhabituelle de caractéristiques")
        
        results.append({
            "image": str(row['image']),
            "anomaly_score": float(round(row['anomaly_score'], 3)),
            "reason": " | ".join(reason_parts),
            "details": {
                "brand": str(row['brand']),
                "country": str(row['country']),
                "age": float(row['age_numeric']),
                "plate": str(row['plate']),
                "damages": [str(d) for d in row['damage']]  # Convertir liste en strings
            }
        })
    
    return results