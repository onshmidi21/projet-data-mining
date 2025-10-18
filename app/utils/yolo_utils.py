# app/utils/yolo_utils.py (version améliorée)
from ultralytics import YOLO
import cv2
import os
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chargement des modèles
try:
    brand_model = YOLO("app/models/logoDetection.pt")
    plate_model = YOLO("app/models/bestV2.pt")
    damage_model = YOLO("app/models/trained.pt")
    logger.info("✅ Tous les modèles YOLO chargés avec succès")
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement des modèles: {e}")
    raise

def detect_brand(image_path):
    """Détecte la marque du véhicule"""
    try:
        if not os.path.exists(image_path):
            logger.warning(f"❌ Fichier introuvable: {image_path}")
            return "Unknown"
        
        result = brand_model.predict(source=image_path, conf=0.5, verbose=False)
        
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
    """Détecte la plaque et retourne l'image cropée de la plaque"""
    try:
        if not os.path.exists(image_path):
            logger.warning(f"❌ Fichier introuvable: {image_path}")
            return None, None
        
        # Charger l'image originale
        original_image = cv2.imread(image_path)
        
        if original_image is None:
            logger.error(f"❌ Impossible de charger l'image: {image_path}")
            return None, None
        
        # Détection avec YOLO
        result = plate_model.predict(source=image_path, conf=0.5, verbose=False)
        
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
                # Crop la région de la plaque
                plate_crop = original_image[y1:y2, x1:x2]
                
                # Sauvegarder l'image cropée
                os.makedirs("runs/plates/cropped", exist_ok=True)
                crop_path = f"runs/plates/cropped/plate_{os.path.basename(image_path)}"
                success = cv2.imwrite(crop_path, plate_crop)
                
                if success:
                    logger.info(f"✅ Plaque détectée et sauvegardée: {crop_path}")
                    return plate_crop, crop_path
                else:
                    logger.error(f"❌ Erreur lors de la sauvegarde: {crop_path}")
                    return plate_crop, None
            else:
                logger.warning("❌ Coordonnées de plaque invalides")
                return None, None
        else:
            logger.info("❌ Aucune plaque détectée")
            return None, None
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la détection de plaque: {e}")
        return None, None

def detect_damage(image_path):
    """Détecte les dommages sur le véhicule"""
    try:
        if not os.path.exists(image_path):
            logger.warning(f"❌ Fichier introuvable: {image_path}")
            return ["No damage"]
        
        result = damage_model.predict(source=image_path, conf=0.5, verbose=False)
        
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