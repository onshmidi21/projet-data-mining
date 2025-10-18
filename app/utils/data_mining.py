import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare les données : encodage, gestion des Unknown, etc."""
    df_clean = df.copy()
    df_clean['damage_type'] = df_clean['damage'].apply(lambda x: 'Damaged' if x != ['No damage'] else 'No damage')
    df_clean['age_numeric'] = pd.to_numeric(df_clean['age'], errors='coerce').fillna(0)
    df_clean = df_clean.dropna(subset=['brand', 'country'])
    return df_clean

def perform_clustering(df: pd.DataFrame) -> dict:
    """Clustering KMeans sur marque, âge et type de dommage."""
    df_clean = prepare_data(df)
    if len(df_clean) < 3:
        return {"error": "Insufficient data for clustering"}
    
    # Features pour clustering
    features = pd.get_dummies(df_clean[['brand', 'damage_type']])
    features['age'] = df_clean['age_numeric']
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=min(3, len(df_clean)), random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    
    df_clean['cluster'] = clusters
    cluster_summary = df_clean.groupby('cluster').agg({
        'brand': 'count',
        'age_numeric': 'mean',
        'damage_type': lambda x: (x == 'Damaged').sum()
    }).round(2).to_dict('index')
    
    return {
        "num_clusters": len(set(clusters)),
        "summary": cluster_summary,
        "interpretation": "Clusters représentent des groupes de similarité (ex: Cluster 0 = Jeunes voitures sans dommages)"
    }

def find_association_rules(df: pd.DataFrame) -> list:
    """Règles d'association Apriori sur marque + âge + dommages."""
    df_clean = prepare_data(df)
    if len(df_clean) < 5:
        return [{"warning": "Insufficient data for associations"}]
    
    # Transactions : combiner marque, bin âge, dommage
    df_clean['age_bin'] = pd.cut(df_clean['age_numeric'], bins=[0, 5, 10, np.inf], labels=['Young', 'Middle', 'Old'], include_lowest=True)
    df_clean = df_clean.dropna(subset=['age_bin'])  # Sécurité supplémentaire contre tout NaN inattendu
    transactions = df_clean[['brand', 'age_bin', 'damage_type']].values.tolist()
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Apriori
    freq_items = apriori(df_trans, min_support=0.1, use_colnames=True)
    if freq_items.empty:
        return [{"warning": "No frequent itemsets found (try lower min_support)"}]
    
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.5)
    rules_formatted = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_dict('records')
    
    return [
        {
            "antecedent": list(rule['antecedents']),
            "consequent": list(rule['consequents']),
            "support": round(rule['support'], 3),
            "confidence": round(rule['confidence'], 3),
            "lift": round(rule['lift'], 3),
            "interpretation": f"Si {list(rule['antecedents'])}, alors {list(rule['consequents'])} (force: {round(rule['confidence'], 2)})"
        }
        for rule in rules_formatted
    ][:5]  # Top 5 règles

def predict_damage_risk(df: pd.DataFrame) -> dict:
    """Random Forest pour prédire risque de dommages."""
    df_clean = prepare_data(df)
    if len(df_clean) < 10:
        return {"error": "Insufficient data for prediction model"}
    
    # Target : 1 si damaged, 0 sinon
    df_clean['target'] = (df_clean['damage_type'] == 'Damaged').astype(int)
    
    # Vérifier si seulement une classe unique
    unique_classes = df_clean['target'].nunique()
    if unique_classes == 1:
        risk_class = df_clean['target'].iloc[0]
        risks = np.full(len(df_clean), 1.0 if risk_class == 1 else 0.0)
        avg_risk = risks.mean()
        high_risk_cars = df_clean[df_clean['target'] == 1][['image', 'target']].rename(columns={'target': 'risk_score'}).to_dict('records')[:3] if risk_class == 1 else []
        return {
            "warning": f"Single class detected ({'Damaged' if risk_class == 1 else 'No Damage'}), risk is uniform.",
            "avg_risk": round(avg_risk, 3),
            "high_risk_cars": high_risk_cars,
            "feature_importance": {}  # Pas de features car pas de fit
        }
    
    # Features
    le_brand = LabelEncoder()
    le_country = LabelEncoder()
    df_clean['brand_encoded'] = le_brand.fit_transform(df_clean['brand'])
    df_clean['country_encoded'] = le_country.fit_transform(df_clean['country'])
    
    X = df_clean[['brand_encoded', 'country_encoded', 'age_numeric']]
    y = df_clean['target']
    
    if len(X) < 2:
        return {"error": "Too few samples for train/test split"}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    
    accuracy = rf.score(X_test, y_test)
    feature_importance = dict(zip(X.columns, rf.feature_importances_.round(3)))
    
    # Prédictions sur tout le dataset
    proba = rf.predict_proba(X)
    if proba.shape[1] == 2:
        risks = proba[:, 1].round(3)
    else:
        # Fallback improbable, mais pour sécurité
        risks = np.full(len(X), 0.5).round(3)
    
    df_clean['risk_score'] = risks
    
    return {
        "accuracy": round(accuracy, 3),
        "feature_importance": feature_importance,
        "avg_risk": round(risks.mean(), 3),
        "high_risk_cars": df_clean[df_clean['risk_score'] > 0.5][['image', 'risk_score']].to_dict('records')[:3]
    }

def detect_anomalies(df: pd.DataFrame) -> list:
    """Isolation Forest pour anomalies (ex: âges/plaques suspects)."""
    df_clean = prepare_data(df)
    if len(df_clean) < 5:
        return [{"warning": "Insufficient data for anomaly detection"}]
    
    # Features pour anomalies : plaque longueur, âge, etc.
    df_clean['plate_length'] = df_clean['plate'].str.len().fillna(0)
    features = df_clean[['age_numeric', 'plate_length']].fillna(0)
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies_pred = iso_forest.fit_predict(features)
    anomalies_df = df_clean[anomalies_pred == -1].copy()
    
    if anomalies_df.empty:
        return [{"message": "No anomalies detected"}]
    
    return [
        {
            "image": row['image'],
            "anomaly_score": abs(iso_forest.decision_function(features.iloc[[i]]))[0] if i < len(features) else 0,
            "reason": f"Âge suspect ({row['age']}) ou plaque anormale (len: {row['plate_length']})",
            "details": row.to_dict()
        }
        for i, (_, row) in enumerate(anomalies_df.iterrows())
    ][:5]