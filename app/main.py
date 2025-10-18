# main.py - Complete FastAPI Application with Model Testing (TorchMetrics Only) - CORRECTED
from app.utils.data_mining import detect_anomalies, find_association_rules, perform_clustering, predict_damage_risk
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import pandas as pd
import numpy as np
from typing import List, Optional
from pydantic import BaseModel
from app.utils.yolo_utils import detect_brand, detect_plate, detect_damage, brand_model, plate_model, damage_model
from app.utils.ocr_utils import extract_text_ocr, extract_country_and_age
import cv2
import base64
import torch
import time
from pathlib import Path

# Import torchmetrics pour les métriques
try:
    from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
    from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    raise ImportError("❌ torchmetrics is required. Install with: pip install torchmetrics")

app = FastAPI(title="Car Analysis Pipeline with Model Testing")
RESULTS = []

# Configuration du device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================== PYDANTIC MODELS ====================

class MiningResult(BaseModel):
    status: str
    clustering: dict
    associations: List[dict]
    predictions: dict
    anomalies: List[dict]

class AnalysisResult(BaseModel):
    status: str
    rows: int
    summary: dict

class DetectionBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str

class SingleImageResult(BaseModel):
    status: str
    image_name: str
    brand: str
    brand_boxes: List[DetectionBox]
    brand_image_base64: Optional[str]
    plate: str
    plate_boxes: List[DetectionBox]
    plate_image_base64: Optional[str]
    plate_cropped_base64: Optional[str]
    country: str
    age: str
    damage: List[str]
    damage_boxes: List[DetectionBox]
    damage_image_base64: Optional[str]

class ModelTestRequest(BaseModel):
    model_type: str  # 'brand', 'plate', ou 'damage'
    dataset_path: str  # Chemin vers le dossier contenant images/ et labels/

class ModelTestResult(BaseModel):
    status: str
    model_type: str
    total_images: int
    processed_images: int
    skipped_images: int
    metrics: dict
    confusion_matrix: List[List[int]]
    class_names: List[str]
    per_class_metrics: dict

# ==================== CORS MIDDLEWARE ====================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== UTILITY FUNCTIONS ====================

def image_to_base64(image):
    """Convertir une image OpenCV en base64"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def draw_boxes_on_image(image_path, boxes, labels, confidences, colors=None):
    """Dessiner les bounding boxes sur l'image"""
    image = cv2.imread(image_path)
    
    if colors is None:
        colors = [(0, 255, 0)] * len(boxes)  # Vert par défaut
    
    for i, (box, label, conf) in enumerate(zip(boxes, labels, confidences)):
        x1, y1, x2, y2 = map(int, box)
        color = colors[i] if i < len(colors) else (0, 255, 0)
        
        # Dessiner le rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Préparer le texte
        text = f"{label}: {conf:.2f}"
        
        # Calculer la taille du texte pour le fond
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Dessiner le fond du texte
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Dessiner le texte
        cv2.putText(
            image,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    return image

def parse_yolo_label(label_path):
    """Parse un fichier label YOLO format (class x y w h)"""
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        annotations = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                annotations.append(class_id)
        
        return annotations
    except Exception as e:
        return []

def compute_metrics_torchmetrics(y_true, y_pred, num_classes, is_single_class_model=False):
    """
    Calcule les métriques en utilisant TORCHMETRICS uniquement
    Version corrigée pour gérer correctement les modèles à classe unique
    """
    # Convertir en tensors PyTorch
    y_true_t = torch.tensor(y_true, dtype=torch.long)
    y_pred_t = torch.tensor(y_pred, dtype=torch.long)
    
    # Déplacer sur le même device que les modèles
    y_true_t = y_true_t.to(device)
    y_pred_t = y_pred_t.to(device)
    
    print(f"\n📊 Calcul des métriques:")
    print(f"   Nombre de classes: {num_classes}")
    print(f"   Distribution y_true: {torch.bincount(y_true_t, minlength=num_classes)}")
    print(f"   Distribution y_pred: {torch.bincount(y_pred_t, minlength=num_classes)}")
    
    # CORRECTION POUR CLASSE UNIQUE
    if is_single_class_model and num_classes == 2:
        print("⚠️  Mode binaire activé (détection présence/absence)")
    
    # Initialiser les métriques avec torchmetrics
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes, average='weighted').to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes, average='weighted').to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average='weighted').to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes, average='weighted').to(device)
    
    precision_per_class_metric = MulticlassPrecision(num_classes=num_classes, average='none').to(device)
    recall_per_class_metric = MulticlassRecall(num_classes=num_classes, average='none').to(device)
    f1_per_class_metric = MulticlassF1Score(num_classes=num_classes, average='none').to(device)
    
    cm_metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    
    # Calculer toutes les métriques
    accuracy = accuracy_metric(y_pred_t, y_true_t)
    precision_weighted = precision_metric(y_pred_t, y_true_t)
    recall_weighted = recall_metric(y_pred_t, y_true_t)
    f1_weighted = f1_metric(y_pred_t, y_true_t)
    
    precision_per_class = precision_per_class_metric(y_pred_t, y_true_t)
    recall_per_class = recall_per_class_metric(y_pred_t, y_true_t)
    f1_per_class = f1_per_class_metric(y_pred_t, y_true_t)
    
    confusion_matrix = cm_metric(y_pred_t, y_true_t)
    
    # Calculer le support (nombre d'instances par classe)
    support = torch.bincount(y_true_t, minlength=num_classes).to(device)
    
    # Macro : moyenne seulement sur les classes avec support > 0
    valid_mask = support > 0
    if valid_mask.sum() > 0:
        precision_macro = torch.mean(precision_per_class[valid_mask]).item()
        recall_macro = torch.mean(recall_per_class[valid_mask]).item()
        f1_macro = torch.mean(f1_per_class[valid_mask]).item()
    else:
        precision_macro = 0.0
        recall_macro = 0.0
        f1_macro = 0.0
    
    print(f"   Support par classe: {support.cpu().tolist()}")
    print(f"   Matrice de confusion:\n{confusion_matrix.cpu().numpy()}")
    
    return {
        'accuracy': accuracy.item(),
        'precision_weighted': precision_weighted.item(),
        'recall_weighted': recall_weighted.item(),
        'f1_weighted': f1_weighted.item(),
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': precision_per_class.cpu().tolist(),
        'recall_per_class': recall_per_class.cpu().tolist(),
        'f1_per_class': f1_per_class.cpu().tolist(),
        'support': support.cpu().tolist(),
        'confusion_matrix': confusion_matrix.cpu().tolist()
    }

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "message": "Car Analysis API with Model Testing & Base64 Bounding Box Visualization!",
        "endpoints": {
            "single_image": "/analyze-single-image-upload/",
            "folder_analysis": "/analyze-folder/",
            "data-mining": "/data-mining/",
            "model_testing": "/test-model-performance/",
            "results": "/results/",
            "download": "/download-results/"
        },
        "device": device,
        "torchmetrics_available": TORCHMETRICS_AVAILABLE
    }

@app.post("/analyze-single-image-upload/", response_model=SingleImageResult)
async def analyze_single_image_upload(file: UploadFile = File(...)):
    """
    API pour analyser UNE SEULE image uploadée avec visualisation des bounding boxes.
    Retourne les images annotées en base64 directement dans la réponse.
    """
    # Vérifier l'extension du fichier
    if not file.filename.lower().endswith((".jpg", ".png", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid image format. Use JPG, PNG, or JPEG")

    # Créer un dossier temporaire
    os.makedirs("temp", exist_ok=True)
    temp_path = f"temp/{file.filename}"
    
    try:
        # Sauvegarder temporairement le fichier
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"\n{'='*50}")
        print(f"📤 Image uploaded: {file.filename}")
        print(f"{'='*50}")

        # 1️⃣ Détection de marque avec bounding boxes
        print("🔍 Detecting brand...")
        brand_result = brand_model.predict(source=temp_path, conf=0.5, verbose=False, device=device)
        
        brand_boxes = []
        brand = "Unknown"
        brand_image_base64 = None
        
        if brand_result[0].boxes and len(brand_result[0].boxes) > 0:
            brand = brand_result[0].names[int(brand_result[0].boxes.cls[0])]
            
            # Extraire les boxes
            boxes_data = []
            labels_data = []
            confs_data = []
            
            for box in brand_result[0].boxes:
                coords = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                class_name = brand_result[0].names[int(box.cls[0])]
                
                brand_boxes.append({
                    "x1": float(coords[0]),
                    "y1": float(coords[1]),
                    "x2": float(coords[2]),
                    "y2": float(coords[3]),
                    "confidence": conf,
                    "class_name": class_name
                })
                
                boxes_data.append(coords)
                labels_data.append(class_name)
                confs_data.append(conf)
            
            # Dessiner les boxes sur l'image
            annotated_image = draw_boxes_on_image(
                temp_path, 
                boxes_data, 
                labels_data, 
                confs_data,
                colors=[(255, 0, 0)] * len(boxes_data)  # Rouge pour marque
            )
            brand_image_base64 = image_to_base64(annotated_image)
        
        print(f"✅ Brand: {brand} ({len(brand_boxes)} detections)")

        # 2️⃣ Détection de plaque avec bounding boxes
        print("🔍 Detecting license plate...")
        plate_result = plate_model.predict(source=temp_path, conf=0.5, verbose=False, device=device)
        
        plate_boxes = []
        plate_text = "Unknown"
        country = "Unknown"
        age = "Unknown"
        plate_image_base64 = None
        plate_cropped_base64 = None
        
        if plate_result[0].boxes and len(plate_result[0].boxes) > 0:
            original_image = cv2.imread(temp_path)
            
            # Extraire les boxes
            boxes_data = []
            labels_data = []
            confs_data = []
            
            for box in plate_result[0].boxes:
                coords = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                plate_boxes.append({
                    "x1": float(coords[0]),
                    "y1": float(coords[1]),
                    "x2": float(coords[2]),
                    "y2": float(coords[3]),
                    "confidence": conf,
                    "class_name": "plate"
                })
                
                boxes_data.append(coords)
                labels_data.append("plate")
                confs_data.append(conf)
            
            # Dessiner les boxes sur l'image
            annotated_image = draw_boxes_on_image(
                temp_path,
                boxes_data,
                labels_data,
                confs_data,
                colors=[(0, 0, 255)] * len(boxes_data)  # Bleu pour plaque
            )
            plate_image_base64 = image_to_base64(annotated_image)
            
            # Crop de la première plaque détectée
            first_box = plate_result[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, first_box)
            height, width = original_image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            if x2 > x1 and y2 > y1:
                plate_crop = original_image[y1:y2, x1:x2]
                plate_cropped_base64 = image_to_base64(plate_crop)
                
                # OCR sur la plaque croppée
                print("🔍 Extracting text from plate...")
                plate_text = extract_text_ocr(plate_crop)
                country, age = extract_country_and_age(plate_text)
                print(f"✅ Plate: {plate_text}, Country: {country}, Age: {age}")

        # 3️⃣ Détection des dommages avec bounding boxes
        print("🔍 Detecting damage...")
        damage_result = damage_model.predict(source=temp_path, conf=0.5, verbose=False, device=device)
        
        damage_boxes = []
        damage = ["No damage"]
        damage_image_base64 = None
        
        if damage_result[0].boxes:
            damage = [damage_result[0].names[int(c)] for c in damage_result[0].boxes.cls]
            
            # Extraire les boxes
            boxes_data = []
            labels_data = []
            confs_data = []
            
            for box in damage_result[0].boxes:
                coords = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                class_name = damage_result[0].names[int(box.cls[0])]
                
                damage_boxes.append({
                    "x1": float(coords[0]),
                    "y1": float(coords[1]),
                    "x2": float(coords[2]),
                    "y2": float(coords[3]),
                    "confidence": conf,
                    "class_name": class_name
                })
                
                boxes_data.append(coords)
                labels_data.append(class_name)
                confs_data.append(conf)
            
            # Dessiner les boxes sur l'image
            annotated_image = draw_boxes_on_image(
                temp_path,
                boxes_data,
                labels_data,
                confs_data,
                colors=[(0, 255, 0)] * len(boxes_data)  # Vert pour dommages
            )
            damage_image_base64 = image_to_base64(annotated_image)
        
        print(f"✅ Damage: {damage} ({len(damage_boxes)} detections)")
        print(f"{'='*50}\n")

        return {
            "status": "success",
            "image_name": file.filename,
            "brand": brand,
            "brand_boxes": brand_boxes,
            "brand_image_base64": brand_image_base64,
            "plate": plate_text,
            "plate_boxes": plate_boxes,
            "plate_image_base64": plate_image_base64,
            "plate_cropped_base64": plate_cropped_base64,
            "country": country,
            "age": age,
            "damage": damage,
            "damage_boxes": damage_boxes,
            "damage_image_base64": damage_image_base64
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    finally:
        # Nettoyer le fichier temporaire
        if os.path.exists(temp_path):
            os.remove(temp_path)

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
        plate_crop = detect_plate(image_path)
        
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
            plate_text = extract_text_ocr(plate_crop)
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

@app.get("/results/")
async def get_results():
    """Endpoint to get current detection results"""
    return {"results": RESULTS}

@app.post("/data-mining/", response_model=MiningResult)
async def run_data_mining():
    """
    API séparée pour le data mining sur les résultats de détection.
    """
    csv_path = "app/data/results.csv"
    
    # Charger depuis CSV si RESULTS vide
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

    # 1. Clustering : Grouper les voitures similaires
    clustering_result = perform_clustering(df)

    # 2. Règles d'association
    associations = find_association_rules(df)

    # 3. Prédictions
    predictions = predict_damage_risk(df)

    # 4. Anomalies
    anomalies = detect_anomalies(df)

    return {
        "status": "success",
        "clustering": clustering_result,
        "associations": associations,
        "predictions": predictions,
        "anomalies": anomalies
    }

@app.post("/test-model-performance/", response_model=ModelTestResult)
async def test_model_performance(request: ModelTestRequest):
    """
    Teste les performances d'un modèle YOLO sur un dataset avec ground truth.
    CORRECTION : Gère correctement les modèles à classe unique en mode binaire.
    
    Structure attendue du dataset:
    dataset_path/
        images/
            img1.jpg
            img2.jpg
            ...
        labels/
            img1.txt  (format YOLO: class_id x_center y_center width height)
            img2.txt
            ...
            
    IMPORTANT pour les modèles à classe unique (ex: détection de plaques):
    - Incluez des images SANS l'objet (labels vides ou absents)
    - Ces images seront traitées comme des exemples négatifs
    """
    
    # Sélectionner le modèle
    if request.model_type == 'brand':
        model = brand_model
        model_name = "Brand Detection"
    elif request.model_type == 'plate':
        model = plate_model
        model_name = "License Plate Detection"
    elif request.model_type == 'damage':
        model = damage_model
        model_name = "Damage Detection"
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type. Use 'brand', 'plate', or 'damage'")
    
    # Obtenir le nombre de classes du modèle
    model_class_names = list(model.names.values())
    model_num_classes = len(model_class_names)
    is_single_class_model = model_num_classes == 1
    
    # Vérifier les chemins
    dataset_path = Path(request.dataset_path)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    if not images_dir.exists():
        raise HTTPException(status_code=404, detail=f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise HTTPException(status_code=404, detail=f"Labels directory not found: {labels_dir}")
    
    # Récupérer toutes les images
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))
    
    if not image_files:
        raise HTTPException(status_code=400, detail="No images found in dataset")
    
    print(f"\n{'='*60}")
    print(f"🧪 Testing {model_name} Model")
    print(f"📊 Dataset: {request.dataset_path}")
    print(f"📸 Total images: {len(image_files)}")
    print(f"🏷️  Model classes: {model_num_classes} - {model_class_names}")
    if is_single_class_model:
        print(f"⚠️  Single-class model detected - Using binary evaluation mode")
    print(f"{'='*60}\n")
    
    # Initialiser les listes pour les prédictions et ground truth
    y_true = []
    y_pred = []
    inference_times = []
    processed_count = 0
    skipped_count = 0
    
    # Compteurs pour debug
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    # Tester sur chaque image
    for i, img_path in enumerate(image_files):
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        # GESTION DES CAS NÉGATIFS : images sans objets
        has_ground_truth = label_path.exists() and label_path.stat().st_size > 0
        
        if has_ground_truth:
            # Charger ground truth
            gt_classes = parse_yolo_label(label_path)
            if not gt_classes:
                has_ground_truth = False
        
        # Prédiction avec timing
        start_time = time.time()
        try:
            result = model.predict(source=str(img_path), conf=0.5, verbose=False, device=device)
            inference_time = (time.time() - start_time) * 1000  # en ms
            inference_times.append(inference_time)
        except Exception as e:
            print(f"❌ Prediction failed for {img_path.name}: {str(e)}")
            skipped_count += 1
            continue
        
        # Extraire les prédictions
        has_prediction = result[0].boxes is not None and len(result[0].boxes) > 0
        
        # LOGIQUE POUR CLASSE UNIQUE (détection binaire)
        if is_single_class_model:
            # Classe 0 = objet présent (True)
            # Classe 1 = objet absent (False)
            
            if has_ground_truth:
                gt_class = 0  # objet présent dans ground truth
            else:
                gt_class = 1  # objet absent dans ground truth
            
            if has_prediction:
                pred_class = 0  # objet détecté
            else:
                pred_class = 1  # objet non détecté
            
            # Comptage pour debug
            if gt_class == 0 and pred_class == 0:
                true_positives += 1
            elif gt_class == 1 and pred_class == 0:
                false_positives += 1
            elif gt_class == 1 and pred_class == 1:
                true_negatives += 1
            elif gt_class == 0 and pred_class == 1:
                false_negatives += 1
        
        else:
            # LOGIQUE MULTI-CLASSES
            if has_ground_truth:
                gt_classes = parse_yolo_label(label_path)
                gt_class = gt_classes[0] if gt_classes else 0
            else:
                gt_class = 0
            
            if has_prediction:
                pred_classes = [int(c) for c in result[0].boxes.cls.cpu().numpy()]
                pred_class = pred_classes[0] if pred_classes else 0
            else:
                pred_class = 0
        
        y_true.append(gt_class)
        y_pred.append(pred_class)
        processed_count += 1
        
        # Debug: afficher quelques exemples
        if i < 5 or (is_single_class_model and i < 10):
            status = '✅' if gt_class == pred_class else '❌'
            print(f"   [{i+1}] {img_path.name}: GT={gt_class}, Pred={pred_class} {status}")
        
        # Afficher la progression
        if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
            print(f"📊 Processed {i + 1}/{len(image_files)} images...")
    
    if not y_true:
        raise HTTPException(status_code=400, detail="No valid image-label pairs found after processing")
    
    print(f"\n✅ Successfully processed {processed_count} images")
    if skipped_count > 0:
        print(f"⚠️  Skipped {skipped_count} images (missing or invalid labels)")
    
    # CORRECTION : Pour un modèle à classe unique, forcer num_classes = 2
    if is_single_class_model:
        num_classes = 2
        class_names = [f"{model_class_names[0]}_Present", f"{model_class_names[0]}_Absent"]
        print(f"\n📊 Binary Classification Mode:")
        print(f"   True Positives (TP):  {true_positives}")
        print(f"   True Negatives (TN):  {true_negatives}")
        print(f"   False Positives (FP): {false_positives}")
        print(f"   False Negatives (FN): {false_negatives}")
        
        # Vérifier si le dataset est déséquilibré
        total_positives = true_positives + false_negatives
        total_negatives = true_negatives + false_positives
        print(f"   Positives: {total_positives}, Negatives: {total_negatives}")
        
        if total_negatives == 0:
            print(f"\n⚠️  WARNING: Aucun exemple négatif détecté!")
            print(f"   Votre dataset ne contient que des images avec l'objet détecté.")
            print(f"   Pour des métriques réalistes, ajoutez des images SANS l'objet!")
    else:
        num_classes = model_num_classes
        class_names = model_class_names
    
    print(f"\n🔢 Nombre de classes pour évaluation: {num_classes}")
    print(f"📏 True labels range: {min(y_true)} to {max(y_true)}")
    print(f"📐 Pred labels range: {min(y_pred)} to {max(y_pred)}")
    
    # Calculer les métriques avec TORCHMETRICS
    try:
        metrics_dict = compute_metrics_torchmetrics(y_true, y_pred, num_classes, is_single_class_model)
        cm_list = metrics_dict.pop('confusion_matrix')
    except Exception as e:
        print(f"❌ Error computing metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error computing metrics: {str(e)}")
    
    # Métriques par classe (seulement pour les classes avec support > 0)
    per_class_metrics = {}
    supports = metrics_dict['support']
    precisions = metrics_dict['precision_per_class']
    recalls = metrics_dict['recall_per_class']
    f1s = metrics_dict['f1_per_class']
    
    for class_id, class_name in enumerate(class_names):
        support = supports[class_id] if class_id < len(supports) else 0
        precision = precisions[class_id] if class_id < len(precisions) else 0.0
        recall = recalls[class_id] if class_id < len(recalls) else 0.0
        f1 = f1s[class_id] if class_id < len(f1s) else 0.0
        
        if support > 0:
            per_class_metrics[class_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "support": int(support)
            }
    
    # Calculer les statistiques de temps d'inférence
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    std_inference_time = np.std(inference_times) if inference_times else 0
    min_inference_time = np.min(inference_times) if inference_times else 0
    max_inference_time = np.max(inference_times) if inference_times else 0
    
    # Afficher les résultats détaillés
    print(f"\n{'='*60}")
    print(f"📊 PERFORMANCE RESULTS - {model_name}")
    print(f"{'='*60}")
    print(f"📈 Accuracy:           {metrics_dict['accuracy']:.4f}")
    print(f"📈 Precision (W):      {metrics_dict['precision_weighted']:.4f}")
    print(f"📈 Recall (W):         {metrics_dict['recall_weighted']:.4f}")
    print(f"📈 F1 Score (W):       {metrics_dict['f1_weighted']:.4f}")
    print(f"📈 Precision (M):      {metrics_dict['precision_macro']:.4f}")
    print(f"📈 Recall (M):         {metrics_dict['recall_macro']:.4f}")
    print(f"📈 F1 Score (M):       {metrics_dict['f1_macro']:.4f}")
    print(f"⏱️  Inference Time:    {avg_inference_time:.2f}ms ± {std_inference_time:.2f}ms")
    print(f"⏱️  Min/Max Time:      {min_inference_time:.2f}ms / {max_inference_time:.2f}ms")
    print(f"📊 Processed Images:   {processed_count}")
    print(f"📊 Total Images:       {len(image_files)}")
    print(f"📊 Success Rate:       {(processed_count/len(image_files))*100:.1f}%")
    print(f"{'='*60}\n")
    
    # Afficher les métriques par classe
    if per_class_metrics:
        print("📋 PER-CLASS METRICS:")
        print("-" * 60)
        for class_name, metrics in per_class_metrics.items():
            print(f"  {class_name:20} | Prec: {metrics['precision']:.3f} | Rec: {metrics['recall']:.3f} | F1: {metrics['f1_score']:.3f} | Support: {metrics['support']}")
        print()
    
    return {
        "status": "success",
        "model_type": model_name,
        "total_images": len(image_files),
        "processed_images": processed_count,
        "skipped_images": skipped_count,
        "metrics": {
            "accuracy": metrics_dict['accuracy'],
            "precision_weighted": metrics_dict['precision_weighted'],
            "recall_weighted": metrics_dict['recall_weighted'],
            "f1_score_weighted": metrics_dict['f1_weighted'],
            "precision_macro": metrics_dict['precision_macro'],
            "recall_macro": metrics_dict['recall_macro'],
            "f1_score_macro": metrics_dict['f1_macro'],
            "inference_time": float(avg_inference_time),
            "inference_time_std": float(std_inference_time),
            "inference_time_min": float(min_inference_time),
            "inference_time_max": float(max_inference_time),
            "success_rate": float(processed_count / len(image_files))
        },
        "confusion_matrix": cm_list,
        "class_names": class_names,
        "per_class_metrics": per_class_metrics
    }

@app.get("/download-results/")
async def download_results():
    """Télécharger le fichier CSV des résultats"""
    csv_path = "app/data/results.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Results file not found")
    return FileResponse(csv_path, media_type='text/csv', filename='results.csv')