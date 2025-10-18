# app/utils/ocr_utils.py - Version améliorée
import cv2
import numpy as np
import os
import re
import easyocr
from typing import Tuple, List, Optional
from datetime import datetime

class WorldwidePlateOCR:
    def __init__(self):
        # Initialiser EasyOCR avec les langues principales
        print("🔄 Initialisation EasyOCR mondial...")
        try:
            # Charger uniquement l'anglais pour couvrir l'alphabet latin
            self.reader = easyocr.Reader(['en'], gpu=False)
            print("✅ EasyOCR initialisé (Latin)")
        except Exception as e:
            print(f"❌ Erreur initialisation EasyOCR: {e}")
            raise
        
        # Dictionnaire COMPLET des patterns de plaques par pays avec noms complets
        self.country_patterns = {
            # ============= EUROPE =============
            'France': [
                r'^[A-Z]{2}[-\s]?\d{3}[-\s]?[A-Z]{2}$',  # AB-123-CD (depuis 2009)
                r'^\d{1,4}[-\s]?[A-Z]{1,3}[-\s]?\d{2}$', # 123-ABC-12 (1950-2009)
                r'^\d{1,3}[-\s]?[A-Z]{1,3}[-\s]?\d{1,2}$', # Anciens formats
            ],
            'Allemagne': [
                r'^[A-Z]{1,3}[-\s]?[A-Z]{1,2}[-\s]?\d{1,4}[A-Z]?$',
            ],
            'Royaume-Uni': [
                r'^[A-Z]{2}\d{2}[-\s]?[A-Z]{3}$',  # AB12-CDE (depuis 2001)
                r'^[A-Z]\d{1,3}[-\s]?[A-Z]{3}$',   # A123-BCD (ancien)
                r'^[A-Z]{3}[-\s]?\d{3}$',          # ABC-123 (très ancien)
            ],
            'Espagne': [
                r'^\d{4}[-\s]?[A-Z]{3}$',
                r'^[A-Z]{1,2}[-\s]?\d{4}[-\s]?[A-Z]{2}$',
            ],
            'Italie': [
                r'^[A-Z]{2}[-\s]?\d{3}[-\s]?[A-Z]{2}$',
            ],
            'Pays-Bas': [
                r'^[A-Z]{2}[-\s]?\d{2}[-\s]?[A-Z]{2}$',
                r'^\d{2}[-\s]?[A-Z]{3}[-\s]?\d{1}$',
            ],
            'Belgique': [
                r'^[A-Z]{3}[-\s]?\d{3}$',
                r'^\d[-\s]?[A-Z]{3}[-\s]?\d{3}$',
            ],
            'Suisse': [
                r'^[A-Z]{2}[-\s]?\d{1,6}$',
            ],
            # ============= AMÉRIQUE =============
            'États-Unis': [
                r'^[A-Z0-9]{2,8}$',
                r'^[A-Z]{3}[-\s]?\d{4}$',
                r'^\d{3}[-\s]?[A-Z]{3}$',
            ],
            'Canada': [
                r'^[A-Z]{4}[-\s]?\d{3}$',
                r'^[A-Z]{3}[-\s]?\d{3}$',
                r'^\d{3}[-\s]?[A-Z]{3}$',
            ],
            # ============= ASIE =============
            'Japon': [
                r'^\d{2,4}[-\s]?[A-Z][-\s]?\d{2,4}$',
            ],
            'Chine': [
                r'^[A-Z]{2}[-\s]?\d{5}$',
            ],
            'Corée du Sud': [
                r'^\d{3}[-\s]?[A-Z][-\s]?\d{4}$',
            ],
        }
        
        # Pattern générique pour plaques non identifiées
        self.generic_pattern = r'^[A-Z0-9]{3,10}$'
        
        # Dictionnaire des âges par pays avec années de changement
        self.age_references = {
            'France': {
                'Très ancien': {'end_year': 1950, 'patterns': [r'^\d{1,3}[A-Z]{1,3}$']},
                'Ancien': {'end_year': 2009, 'patterns': [
                    r'^\d{1,4}[-\s]?[A-Z]{1,3}[-\s]?\d{2}$',
                    r'^\d{1,3}[A-Z]{1,3}\d{1,2}$'
                ]},
                'Moderne': {'start_year': 2009, 'patterns': [
                    r'^[A-Z]{2}[-\s]?\d{3}[-\s]?[A-Z]{2}$',
                    r'^[A-Z]{2}\d{3}[A-Z]{2}$'
                ]}
            },
            'Royaume-Uni': {
                'Ancien': {'end_year': 2001, 'patterns': [
                    r'^[A-Z]\d{1,3}[-\s]?[A-Z]{3}$',
                    r'^[A-Z]{3}[-\s]?\d{3}$'
                ]},
                'Moderne': {'start_year': 2001, 'patterns': [
                    r'^[A-Z]{2}\d{2}[-\s]?[A-Z]{3}$',
                    r'^[A-Z]{2}\d{2}[A-Z]{3}$'
                ]}
            },
            'Allemagne': {
                'Ancien': {'end_year': 1990, 'patterns': [
                    r'^[A-Z]{1,2}[-\s]?\d{1,3}$'
                ]},
                'Moderne': {'start_year': 1990, 'patterns': [
                    r'^[A-Z]{1,3}[-\s]?[A-Z]{1,2}[-\s]?\d{1,4}[A-Z]?$'
                ]}
            },
            'États-Unis': {
                'Ancien': {'end_year': 1990, 'patterns': [
                    r'^\d{3}[A-Z]{3}$',
                    r'^[A-Z]{3}\d{3}$'
                ]},
                'Moderne': {'start_year': 1990, 'patterns': [
                    r'^[A-Z0-9]{6,8}$',
                    r'^[A-Z]{3}[-\s]?\d{4}$'
                ]}
            }
        }

    def preprocess_plate_image(self, plate_image):
        """Prétraite l'image de plaque"""
        try:
            if isinstance(plate_image, str):
                if not os.path.exists(plate_image):
                    return None
                image = cv2.imread(plate_image)
            else:
                image = plate_image
            
            if image is None:
                return None
            
            # Redimensionner
            height, width = image.shape[:2]
            target_width = 600
            scale = target_width / width
            new_w, new_h = int(width * scale), int(height * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Niveaux de gris
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Amélioration contraste
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Débruitage
            denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
            
            # Sauvegarder debug
            debug_dir = "runs/plates/debug"
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, "preprocessed.jpg")
            cv2.imwrite(debug_path, denoised)
            
            return denoised
            
        except Exception as e:
            print(f"❌ Erreur prétraitement: {e}")
            return None

    def extract_text_universal(self, plate_image):
        """Extraction universelle"""
        try:
            processed = self.preprocess_plate_image(plate_image)
            if processed is None:
                print("❌ Prétraitement échoué")
                return "Unknown"
            
            # Essayer OCR avec le reader principal
            results = self._try_ocr_with_reader(processed, self.reader)
            
            if not results:
                print("❌ Aucun texte détecté")
                return "Unknown"
            
            # Extraire textes avec bonne confiance
            detected_texts = []
            for detection in results:
                text = detection[1]
                confidence = detection[2]
                print(f"📝 Détecté: '{text}' (confiance: {confidence:.2f})")
                
                if confidence > 0.3:  # Seuil de confiance augmenté
                    detected_texts.append((text, confidence))
            
            if not detected_texts:
                return "Unknown"
            
            # Trier par confiance
            detected_texts.sort(key=lambda x: x[1], reverse=True)
            
            # Essayer chaque texte détecté
            for text, confidence in detected_texts:
                cleaned = self.clean_universal_text(text)
                print(f"🧹 Nettoyé: '{cleaned}'")
                
                if len(cleaned) >= 4:  # Longueur minimale augmentée
                    country = self.identify_country(cleaned)
                    print(f"🌍 Pays identifié: {country}")
                    
                    if country != "Inconnu":
                        return cleaned
            
            # Retourner meilleur texte même si pays inconnu
            best_text = self.clean_universal_text(detected_texts[0][0])
            if len(best_text) >= 4:
                print(f"⚠️ Format non standard: {best_text}")
                return best_text
            
            return "Unknown"
            
        except Exception as e:
            print(f"❌ Erreur OCR: {e}")
            import traceback
            traceback.print_exc()
            return "Unknown"

    def _try_ocr_with_reader(self, image, reader):
        """Essaie OCR avec un reader spécifique"""
        try:
            return reader.readtext(
                image,
                detail=1,
                paragraph=False,
                batch_size=4,
                min_size=20,  # Taille minimale augmentée
                contrast_ths=0.3,
                adjust_contrast=0.7,
                text_threshold=0.5
            )
        except Exception as e:
            print(f"⚠️ Erreur avec reader: {e}")
            return []

    def clean_universal_text(self, text):
        """Nettoie le texte de manière plus agressive"""
        # Convertir en majuscules
        text = text.upper()
        
        # Supprimer les caractères spéciaux indésirables
        text = re.sub(r'[^A-Z0-9\-\s]', '', text)
        
        # Remplacer multiples espaces/tirets par un seul
        text = re.sub(r'[-\s]+', '', text)
        
        # Corrections courantes
        common_errors = {
            '0': 'O', '1': 'I', '5': 'S', '8': 'B'
        }
        
        # Appliquer corrections si le texte est court
        if len(text) <= 8:
            corrected = ''
            for char in text:
                corrected += common_errors.get(char, char)
            text = corrected
        
        return text

    def identify_country(self, plate_text):
        """Identifie le pays avec nom complet"""
        for country, patterns in self.country_patterns.items():
            for pattern in patterns:
                if re.match(pattern, plate_text, re.IGNORECASE):
                    return country
        
        # Vérifier le pattern générique
        if re.match(self.generic_pattern, plate_text):
            return "International"
        
        return "Inconnu"

    def determine_age(self, plate_text, country):
        """Détermine l'âge de la plaque avec plus de précision"""
        if country == "Inconnu" or country == "International":
            return "Indéterminé"
        
        current_year = datetime.now().year
        
        # Vérifier les références d'âge par pays
        if country in self.age_references:
            age_categories = self.age_references[country]
            
            for age_name, age_info in age_categories.items():
                for pattern in age_info['patterns']:
                    if re.match(pattern, plate_text):
                        # Affiner avec les années si disponibles
                        if 'end_year' in age_info:
                            if current_year - age_info['end_year'] > 20:
                                return f"{age_name} (avant {age_info['end_year']})"
                            else:
                                return age_name
                        elif 'start_year' in age_info:
                            years_in_use = current_year - age_info['start_year']
                            if years_in_use < 5:
                                return f"{age_name} (récent)"
                            else:
                                return f"{age_name} (depuis {age_info['start_year']})"
                        return age_name
        
        # Méthode de fallback basée sur la structure
        return self._estimate_age_from_structure(plate_text, country)

    def _estimate_age_from_structure(self, plate_text, country):
        """Estime l'âge basé sur la structure de la plaque"""
        current_year = datetime.now().year
        
        # Règles générales par structure
        if re.match(r'^\d{1,4}[A-Z]{1,3}\d{0,2}$', plate_text):
            # Format numérique-alphanumérique (souvent ancien)
            return "Ancien"
        elif re.match(r'^[A-Z]{2}\d{3}[A-Z]{2}$', plate_text):
            # Format alphanumérique équilibré (souvent moderne)
            return "Moderne"
        elif re.match(r'^[A-Z0-9]{6,8}$', plate_text):
            # Format mixte long
            return "Contemporain"
        else:
            # Analyse basée sur la longueur et composition
            if len(plate_text) <= 6:
                return "Classique"
            elif len(plate_text) >= 7:
                return "Récent"
        
        return "Standard"

# Instance globale
worldwide_ocr = WorldwidePlateOCR()

def extract_text_ocr(plate_image):
    """Fonction principale d'extraction OCR"""
    try:
        return worldwide_ocr.extract_text_universal(plate_image)
    except Exception as e:
        print(f"❌ Erreur OCR: {e}")
        return "Unknown"

def extract_country_and_age(plate_text):
    """Extrait pays (nom complet) et âge"""
    if plate_text == "Unknown" or not plate_text or len(plate_text) < 4:
        return "Inconnu", "Indéterminé"
    
    country = worldwide_ocr.identify_country(plate_text)
    age = worldwide_ocr.determine_age(plate_text, country)
    
    return country, age

# Fonction utilitaire pour afficher les informations complètes
def get_complete_plate_info(plate_image):
    """Obtient toutes les informations sur la plaque"""
    plate_text = extract_text_ocr(plate_image)
    country, age = extract_country_and_age(plate_text)
    
    return {
        'plate_text': plate_text,
        'country': country,
        'age': age,
        'confidence': 'high' if plate_text != "Unknown" else 'low'
    }