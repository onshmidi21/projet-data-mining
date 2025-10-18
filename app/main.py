from fastapi import FastAPI, HTTPException
import os
import pandas as pd
from typing import List
from pydantic import BaseModel
from app.utils.yolo_utils import detect_brand, detect_plate, detect_damage
from app.utils.ocr_utils import extract_text_ocr, extract_country_and_age
from app.utils.data_mining import (
    perform_clustering, 
    find_association_rules, 
    predict_damage_risk,
    detect_anomalies
)

app = FastAPI(title="Car Analysis Pipeline")

RESULTS = []

class AnalysisResult(BaseModel):
    status: str
    rows: int
    summary: dict

class MiningResult(BaseModel):
    status: str
    clustering: dict
    associations: List[dict]
    predictions: dict
    anomalies: List[dict]

@app.post("/analyze-folder/", response_model=AnalysisResult)
async def analyze_folder(folder_path: str):
    """
    API pour analyser un dossier d'images : détection de marque, plaque, pays/âge, et dommages.
    Sauvegarde les résultats en CSV et retourne un résumé.
    """
    # Vérifier que le dossier existe
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Folder not found")
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    if not image_files:
        raise HTTPException(status_code=400, detail="No images found in folder")

    RESULTS.clear()  # Réinitialiser les résultats

    for image in image_files:
        image_path = os.path.join(folder_path, image)
        print(f"\n{'='*50}")
        print(f"🔄 Processing: {image}")
        print(f"{'='*50}")

        # 1️⃣ Détection de marque
        print("🔍 Detecting brand...")
        brand = detect_brand(image_path)
        print(f"✅ Brand: {brand}")

        # 2️⃣ Détection de plaque
        print("🔍 Detecting license plate...")
        plate_crop, plate_path = detect_plate(image_path)
        
        # Vérifier si une plaque a été détectée
        if plate_crop is None:
            print("❌ No license plate detected")
            plate_text = "Unknown"
            country = "Unknown"
            age = "Unknown"
        else:
            print("✅ License plate detected and cropped")
            
            # 3️⃣ OCR : extraction du texte de la plaque
            print("🔍 Extracting text from plate...")
            plate_text = extract_text_ocr(plate_crop)  # Utiliser l'image cropée
            country, age = extract_country_and_age(plate_text)
            print(f"✅ Plate: {plate_text}, Country: {country}, Age: {age}")

        # 4️⃣ Détection des dommages
        print("🔍 Detecting damage...")
        damage = detect_damage(image_path)
        print(f"✅ Damage: {damage}")

        # 5️⃣ Enregistrer les résultats
        result = {
            "image": image,
            "brand": brand,
            "plate": plate_text,
            "country": country,
            "age": age,
            "damage": damage
        }
        RESULTS.append(result)
        print(f"💾 Results saved for {image}")

    # Créer un DataFrame et sauvegarder
    df = pd.DataFrame(RESULTS)
    os.makedirs("app/data", exist_ok=True)
    df.to_csv("app/data/results.csv", index=False)

    return {
        "status": "success", 
        "rows": len(df), 
        "summary": {
            "total_images": len(image_files),
            "plates_detected": len([r for r in RESULTS if r['plate'] != 'Unknown']),
            "brands_detected": len([r for r in RESULTS if r['brand'] != 'Unknown']),
            "damage_detected": len([r for r in RESULTS if r['damage'] != ["No damage"]])
        }
    }

@app.post("/data-mining/", response_model=MiningResult)
async def run_data_mining():
    """
    API séparée pour le data mining sur les résultats de détection.
    Utilise des algorithmes créatifs :
    - Clustering (KMeans) pour grouper les voitures par marque/âge/dommages.
    - Règles d'association (Apriori) pour trouver des patterns comme 'marque X -> dommages Y'.
    - Classification (Random Forest) pour prédire le risque de dommages basé sur âge/pays/marque.
    - Détection d'anomalies (Isolation Forest) pour identifier des plaques ou âges suspects.
    """
    csv_path = "app/data/results.csv"
    
    # Charger depuis CSV si RESULTS vide (ex: après redémarrage du serveur)
    if not RESULTS:
        if os.path.exists(csv_path):
            df_loaded = pd.read_csv(csv_path)
            RESULTS.extend(df_loaded.to_dict('records'))
            print(f"📂 Chargé {len(RESULTS)} résultats depuis {csv_path}")
        else:
            raise HTTPException(status_code=400, detail="No detection results available. Run /analyze-folder/ first.")

    df = pd.DataFrame(RESULTS)
    if df.empty:
        raise HTTPException(status_code=400, detail="No data to analyze.")

    print("🔍 Running creative data mining analysis...")

    # 1. Clustering : Grouper les voitures similaires (marque, âge, type de dommage)
    clustering_result = perform_clustering(df)

    # 2. Règles d'association : Patterns comme "Renault + >10 ans -> Rayures fréquentes"
    associations = find_association_rules(df)

    # 3. Prédictions : Risque de dommages basé sur features
    predictions = predict_damage_risk(df)

    # 4. Anomalies : Détecter plaques ou âges incohérents
    anomalies = detect_anomalies(df)

    return {
        "status": "success",
        "clustering": clustering_result,
        "associations": associations,
        "predictions": predictions,
        "anomalies": anomalies
    }

@app.get("/results/")
async def get_results():
    """Endpoint to get current detection results"""
    return {"results": RESULTS}

@app.get("/")
async def root():
    return {"message": "Car Analysis API is running! Separate endpoints for detection and mining."}