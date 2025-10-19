import pandas as pd
import numpy as np
from datetime import datetime

# ÉTAPE 1 : NETTOYAGE ET PRÉPARATION DES DONNÉES

def prepare_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage complet des données :
    - Gestion des valeurs manquantes
    - Encodage des dommages en valeurs numériques
    - Création des colonnes : damage_cost, has_damage
    """
    
    df_clean = df.copy()
    
    # Dictionnaire d'encodage des dommages
    damage_mapping = {
        'scratch': 1,
        'dent': 2,
        'rust': 3,
        'broken_lamp': 4,
        'shattered_glass': 5,
        'no damage': 0,
        'No damage': 0,
        np.nan: 0
    }
    
    # Fonction pour traiter la colonne damage
    def encode_damages(damage_value):
        """Convertir la liste de dommages en valeurs numériques"""
        if pd.isna(damage_value) or damage_value == '' or damage_value == 'No damage':
            return 0
        
        if isinstance(damage_value, list):
            # Si c'est une liste, prendre le dommage le plus grave
            encoded = [damage_mapping.get(str(d).lower(), 0) for d in damage_value]
            return max(encoded) if encoded else 0
        
        # Si c'est une string, chercher dans le mapping
        return damage_mapping.get(str(damage_value).lower(), 0)
    
    # Appliquer l'encodage
    df_clean['damage_encoded'] = df_clean['damage'].apply(encode_damages)
    
    # COLONNE 1 : Coût du dommage (estimation simple basée sur type)
    cost_mapping = {
        0: 0,      # Pas de dommage
        1: 150,    # Scratch
        2: 300,    # Dent
        3: 500,    # Rouille
        4: 200,    # Lampe cassée
        5: 400     # Vitre brisée
    }
    df_clean['damage_cost'] = df_clean['damage_encoded'].map(cost_mapping)
    
    # COLONNE 2 : Indicateur has_damage (Oui/Non)
    df_clean['has_damage'] = df_clean['damage_encoded'].apply(
        lambda x: 'Oui' if x > 0 else 'Non'
    )
    
    # Gestion des valeurs manquantes
    df_clean['brand'] = df_clean['brand'].fillna('Unknown')
    df_clean['country'] = df_clean['country'].fillna('Unknown')
    df_clean['age'] = df_clean['age'].fillna(0)
    
    # Supprimer les lignes complètement vides
    df_clean = df_clean.dropna(subset=['image', 'plate'], how='all')
    
    return df_clean

# ÉTAPE 2 : ANALYSE DESCRIPTIVE

def descriptive_analysis_by_brand(df: pd.DataFrame) -> dict:
    """Analyse descriptive par marque"""
    
    analysis = {}
    
    for brand in df['brand'].unique():
        brand_data = df[df['brand'] == brand]
        
        analysis[str(brand)] = {
            'nombre_vehicules': len(brand_data),
            'vehicules_endommages': int((brand_data['has_damage'] == 'Oui').sum()),
            'taux_dommage_%': round((brand_data['has_damage'] == 'Oui').sum() / len(brand_data) * 100, 2),
            'cout_moyen_dommage': round(brand_data['damage_cost'].mean(), 2),
            'cout_total': round(brand_data['damage_cost'].sum(), 2),
            'age_moyen': round(brand_data['age'].mean(), 2)
        }
    
    return analysis


def descriptive_analysis_by_country(df: pd.DataFrame) -> dict:
    """Analyse descriptive par pays"""
    
    analysis = {}
    
    for country in df['country'].unique():
        country_data = df[df['country'] == country]
        
        analysis[str(country)] = {
            'nombre_vehicules': len(country_data),
            'vehicules_endommages': int((country_data['has_damage'] == 'Oui').sum()),
            'taux_dommage_%': round((country_data['has_damage'] == 'Oui').sum() / len(country_data) * 100, 2),
            'cout_moyen_dommage': round(country_data['damage_cost'].mean(), 2),
            'cout_total': round(country_data['damage_cost'].sum(), 2),
            'age_moyen': round(country_data['age'].mean(), 2)
        }
    
    return analysis


def descriptive_analysis_by_damage_type(df: pd.DataFrame) -> dict:
    """Analyse descriptive par type de dommage"""
    
    damage_types = {
        0: 'Pas de dommage',
        1: 'Rayure',
        2: 'Bosselure',
        3: 'Rouille',
        4: 'Lampe cassée',
        5: 'Vitre brisée'
    }
    
    analysis = {}
    
    for damage_code, damage_name in damage_types.items():
        damage_data = df[df['damage_encoded'] == damage_code]
        
        if len(damage_data) > 0:
            analysis[damage_name] = {
                'nombre_occurrences': len(damage_data),
                'pourcentage_%': round(len(damage_data) / len(df) * 100, 2),
                'cout_unitaire_moyen': round(damage_data['damage_cost'].mean(), 2),
                'cout_total': round(damage_data['damage_cost'].sum(), 2),
                'marques_affectees': list(damage_data['brand'].unique())
            }
    
    return analysis


# ÉTAPE 3 : RAPPORT GLOBAL

def generate_comprehensive_report(df: pd.DataFrame) -> str:
    """Génère un rapport complet et clair"""
    
    df_clean = prepare_and_clean_data(df)
    
    report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    RAPPORT D'ANALYSE - DOMMAGES AUTOMOBILES                  ║
║                              {datetime.now().strftime('%d/%m/%Y %H:%M')}                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

1️⃣  RÉSUMÉ EXÉCUTIF
{'─' * 80}
   • Nombre total de véhicules : {len(df_clean)}
   • Véhicules endommagés : {(df_clean['has_damage'] == 'Oui').sum()} ({round((df_clean['has_damage'] == 'Oui').sum() / len(df_clean) * 100, 2)}%)
   • Véhicules en bon état : {(df_clean['has_damage'] == 'Non').sum()} ({round((df_clean['has_damage'] == 'Non').sum() / len(df_clean) * 100, 2)}%)
   • Coût total des dommages : {df_clean['damage_cost'].sum():.2f}€
   • Coût moyen par dommage : {df_clean[df_clean['damage_cost'] > 0]['damage_cost'].mean():.2f}€
   • Âge moyen des véhicules : {df_clean['age'].mean():.1f} ans

2️⃣  ANALYSE PAR MARQUE
{'─' * 80}
"""
    
    brand_analysis = descriptive_analysis_by_brand(df_clean)
    for brand, stats in sorted(brand_analysis.items(), 
                               key=lambda x: x[1]['taux_dommage_%'], 
                               reverse=True):
        report += f"""
   {brand}:
      └─ Véhicules : {stats['nombre_vehicules']} 
         └─ Endommagés : {stats['vehicules_endommages']} ({stats['taux_dommage_%']}%)
         └─ Coût moyen : {stats['cout_moyen_dommage']}€
         └─ Coût total : {stats['cout_total']}€
         └─ Âge moyen : {stats['age_moyen']} ans
"""
    
    report += f"""
3️⃣  ANALYSE PAR PAYS
{'─' * 80}
"""
    
    country_analysis = descriptive_analysis_by_country(df_clean)
    for country, stats in sorted(country_analysis.items(), 
                                 key=lambda x: x[1]['taux_dommage_%'], 
                                 reverse=True):
        report += f"""
   {country}:
      └─ Véhicules : {stats['nombre_vehicules']}
         └─ Endommagés : {stats['vehicules_endommages']} ({stats['taux_dommage_%']}%)
         └─ Coût moyen : {stats['cout_moyen_dommage']}€
         └─ Coût total : {stats['cout_total']}€
         └─ Âge moyen : {stats['age_moyen']} ans
"""
    
    report += f"""
4️⃣  ANALYSE PAR TYPE DE DOMMAGE
{'─' * 80}
"""
    
    damage_analysis = descriptive_analysis_by_damage_type(df_clean)
    for damage_type, stats in sorted(damage_analysis.items(), 
                                     key=lambda x: x[1]['nombre_occurrences'], 
                                     reverse=True):
        report += f"""
   {damage_type}:
      └─ Occurrences : {stats['nombre_occurrences']} ({stats['pourcentage_%']}%)
         └─ Coût unitaire moyen : {stats['cout_unitaire_moyen']}€
         └─ Coût total : {stats['cout_total']}€
         └─ Marques affectées : {', '.join(stats['marques_affectees'])}
"""
    
    report += f"""
5️⃣  STATISTIQUES DÉTAILLÉES
{'─' * 80}
   Distribution des dommages :
      └─ Minimum : {df_clean['damage_cost'].min():.2f}€
      └─ Maximum : {df_clean['damage_cost'].max():.2f}€
      └─ Médiane : {df_clean['damage_cost'].median():.2f}€
      └─ Écart-type : {df_clean['damage_cost'].std():.2f}€

   Valeurs manquantes :
      └─ Brand : {df_clean['brand'].isna().sum()}
      └─ Country : {df_clean['country'].isna().sum()}
      └─ Age : {df_clean['age'].isna().sum()}
      └─ Damage : {df_clean['damage'].isna().sum()}

╚══════════════════════════════════════════════════════════════════════════════╝
"""
    
    return report

