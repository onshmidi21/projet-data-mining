# app/utils/yolo_utils.py (version améliorée)
from ultralytics import YOLO
import cv2
import os
import logging
import torch

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vérifier la disponibilité du GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"🖥️  Utilisation du device: {device}")

# Chargement des modèles avec GPU si disponible
try:
    brand_model = YOLO("app/models/logoDetection.pt")
    plate_model = YOLO("app/models/bestV2.pt")
    damage_model = YOLO("app/models/trained.pt")
    
    # Déplacer les modèles sur GPU si disponible
    if device == 'cuda':
        brand_model.to(device)
        plate_model.to(device)
        damage_model.to(device)
        logger.info("✅ Tous les modèles YOLO chargés sur GPU avec succès")
    else:
        logger.info("✅ Tous les modèles YOLO chargés sur CPU avec succès")
        
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement des modèles: {e}")
    raise

def detect_brand(image_path):
    """Détecte la marque du véhicule"""
    try:
        if not os.path.exists(image_path):
            logger.warning(f"❌ Fichier introuvable: {image_path}")
            return "Unknown"
        
        result = brand_model.predict(source=image_path, conf=0.5, verbose=False, device=device)
        
        if result[0].boxes and len(result[0].boxes) > 0:
            brand = result[0].names[int(result[0].boxes.cls[0])]
            logger.info(f"✅ Marque détectée: {brand}")
            return brand
        else:
            logger.info("❌ Aucune marque détectée")
            return "Unknown"
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la détection de marque: {e}")
        return "Unknown"

def detect_plate(image_path):
    """Détecte la plaque et retourne l'image cropée de la plaque (sans sauvegarde)"""
    try:
        if not os.path.exists(image_path):
            logger.warning(f"❌ Fichier introuvable: {image_path}")
            return None
        
        # Charger l'image originale
        original_image = cv2.imread(image_path)
        
        if original_image is None:
            logger.error(f"❌ Impossible de charger l'image: {image_path}")
            return None
        
        # Détection avec YOLO sur GPU
        result = plate_model.predict(source=image_path, conf=0.5, verbose=False, device=device)
        
        if result[0].boxes and len(result[0].boxes) > 0:
            # Récupérer les coordonnées de la première détection
            boxes = result[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, boxes)
            
            # S'assurer que les coordonnées sont dans les limites de l'image
            height, width = original_image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            # Vérifier que la région cropée est valide
            if x2 > x1 and y2 > y1:
                # Crop la région de la plaque (sans sauvegarde)
                plate_crop = original_image[y1:y2, x1:x2]
                logger.info(f"✅ Plaque détectée - Dimensions: {plate_crop.shape}")
                return plate_crop
            else:
                logger.warning("❌ Coordonnées de plaque invalides")
                return None
        else:
            logger.info("❌ Aucune plaque détectée")
            return None
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la détection de plaque: {e}")
        return None

def detect_damage(image_path):
    """Détecte les dommages sur le véhicule"""
    try:
        if not os.path.exists(image_path):
            logger.warning(f"❌ Fichier introuvable: {image_path}")
            return ["No damage"]
        
        result = damage_model.predict(source=image_path, conf=0.5, verbose=False, device=device)
        
        if result[0].boxes:
            damages = [result[0].names[int(c)] for c in result[0].boxes.cls]
            logger.info(f"✅ Dommages détectés: {damages}")
            return damages
        else:
            logger.info("✅ Aucun dommage détecté")
            return ["No damage"]
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la détection de dommages: {e}")
        return ["No damage"]