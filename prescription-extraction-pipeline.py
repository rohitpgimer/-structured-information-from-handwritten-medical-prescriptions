import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, LSTM, Bidirectional, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import pandas as pd
import spacy
from transformers import TFAutoModelForTokenClassification, AutoTokenizer

class PrescriptionExtractor:
    def __init__(self, model_paths=None):
        """
        Initialize the prescription extraction pipeline
        
        Args:
            model_paths (dict): Paths to pretrained models for each component
        """
        self.models = {}
        self.tokenizer = None
        self.drug_database = self._load_drug_database()
        
        # Load models if provided
        if model_paths:
            if 'segmentation' in model_paths:
                self.models['segmentation'] = load_model(model_paths['segmentation'])
            if 'ocr' in model_paths:
                self.models['ocr'] = load_model(model_paths['ocr'])
            if 'ner' in model_paths:
                self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
                self.models['ner'] = TFAutoModelForTokenClassification.from_pretrained(model_paths['ner'])
    
    def _load_drug_database(self):
        """Load drug database for validation and correction"""
        # In a real implementation, this would load from a proper medical database
        # For demonstration, we're using a simplified dictionary
        return {
            "common_drugs": [
                "metformin", "lisinopril", "amlodipine", "simvastatin", "omeprazole",
                "albuterol", "gabapentin", "losartan", "levothyroxine", "atorvastatin"
            ],
            "dosage_units": ["mg", "mcg", "ml", "g", "units"],
            "frequency": ["daily", "twice daily", "three times daily", "weekly", "as needed"]
        }
    
    def preprocess_image(self, image):
        """
        Preprocess prescription image for better extraction
        
        Args:
            image: Input prescription image (path or numpy array)
            
        Returns:
            Preprocessed image ready for segmentation
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to deal with varying lighting
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Noise removal
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Deskew (straighten) image if needed
        # This is a simplified version - a more robust approach would be needed in production
        coords = np.column_stack(np.where(binary > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        # Rotate if significant skew detected
        if abs(angle) > 0.5:
            (h, w) = binary.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            binary = cv2.warpAffine(binary, M, (w, h), 
                                    flags=cv2.INTER_CUBIC, 
                                    borderMode=cv2.BORDER_REPLICATE)
        
        # Normalize for neural network input
        normalized = binary / 255.0
        return normalized
    
    def segment_prescription(self, preprocessed_image):
        """
        Segment prescription into regions (patient info, medication, doctor info)
        
        Args:
            preprocessed_image: Preprocessed prescription image
            
        Returns:
            Dictionary of segmented regions
        """
        if 'segmentation' in self.models:
            # Use the trained segmentation model
            regions = self.models['segmentation'].predict(
                np.expand_dims(preprocessed_image, axis=0)
            )
            # Process segmentation output
            # This would extract the actual regions based on model output
            # Simplified for demonstration
            return self._process_segmentation_output(regions, preprocessed_image)
        else:
            # Fallback to rule-based segmentation if no model is available
            return self._rule_based_segmentation(preprocessed_image)
    
    def _rule_based_segmentation(self, preprocessed_image):
        """Simple rule-based segmentation as fallback"""
        h, w = preprocessed_image.shape
        
        # Simple heuristic division - in real implementation, this would be
        # much more sophisticated based on layout analysis
        regions = {
            'header': preprocessed_image[0:int(h*0.2), :],
            'patient_info': preprocessed_image[int(h*0.2):int(h*0.35), :],
            'medication': preprocessed_image[int(h*0.35):int(h*0.8), :],
            'doctor_info': preprocessed_image[int(h*0.8):, :]
        }
        
        return regions
    
    def _process_segmentation_output(self, model_output, image):
        """Process model segmentation output into regions"""
        # This would extract actual regions based on segmentation masks
        # Simplified for demonstration
        return self._rule_based_segmentation(image)
    
    def perform_ocr(self, regions):
        """
        Perform OCR on segmented regions
        
        Args:
            regions: Dictionary of segmented image regions
            
        Returns:
            Dictionary of extracted text by region
        """
        extracted_text = {}
        
        for region_name, region_image in regions.items():
            if 'ocr' in self.models:
                # Use deep learning OCR model
                # Prepare the image for the model
                resized = cv2.resize(region_image, (128, 32))
                text = self._ocr_model_predict(resized)
                extracted_text[region_name] = text
            else:
                # Fallback to external OCR (would use Tesseract in real implementation)
                extracted_text[region_name] = f"Placeholder text for {region_name}"
        
        return extracted_text
    
    def _ocr_model_predict(self, image):
        """Use OCR model to predict text from image"""
        # In a real implementation, this would use the actual OCR model
        # Simplified for demonstration
        return "Sample extracted text"
    
    def extract_entities(self, text_dict):
        """
        Extract medical entities from OCR text
        
        Args:
            text_dict: Dictionary of extracted text by region
            
        Returns:
            Structured data with medical entities
        """
        structured_data = {
            'patient': {},
            'medication': [],
            'doctor': {},
            'date': None,
            'refills': None
        }
        
        # Process patient info
        if 'patient_info' in text_dict:
            structured_data['patient'] = self._extract_patient_info(text_dict['patient_info'])
            
        # Process medication
        if 'medication' in text_dict:
            structured_data['medication'] = self._extract_medication_info(text_dict['medication'])
            
        # Process doctor info
        if 'doctor_info' in text_dict:
            structured_data['doctor'] = self._extract_doctor_info(text_dict['doctor_info'])
            
        # Extract date and refills
        for section, text in text_dict.items():
            # Look for date patterns
            date_info = self._extract_date(text)
            if date_info and not structured_data['date']:
                structured_data['date'] = date_info
                
            # Look for refill information
            refill_info = self._extract_refills(text)
            if refill_info and not structured_data['refills']:
                structured_data['refills'] = refill_info
                
        return structured_data
    
    def _extract_patient_info(self, text):
        """Extract patient information from text"""
        # In a real implementation, this would use NER model or regex patterns
        # to extract name, age, ID, etc.
        return {
            'name': self._extract_name(text),
            'age': self._extract_age(text),
            'gender': self._extract_gender(text),
            'id': self._extract_id(text)
        }
    
    def _extract_medication_info(self, text):
        """Extract medication details from text"""
        if 'ner' in self.models:
            # Use NER model to extract medication entities
            return self._extract_med_entities_with_model(text)
        else:
            # Simplified rule-based extraction
            medications = []
            
            # Split text into potential medication entries
            lines = text.split('\n')
            current_med = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this line might be a drug name
                if any(drug in line.lower() for drug in self.drug_database['common_drugs']):
                    # Save previous med if exists
                    if current_med:
                        medications.append(current_med)
                    
                    # Start new medication
                    current_med = {'name': line}
                
                # Check if this line contains dosage info
                elif current_med and any(unit in line.lower() for unit in self.drug_database['dosage_units']):
                    current_med['dosage'] = line
                
                # Check if this line contains frequency
                elif current_med and any(freq in line.lower() for freq in self.drug_database['frequency']):
                    current_med['frequency'] = line
                
                # Other lines might be instructions
                elif current_med:
                    current_med['instructions'] = line
            
            # Add the last medication if exists
            if current_med:
                medications.append(current_med)
                
            return medications
    
    def _extract_med_entities_with_model(self, text):
        """Use NER model to extract medication entities"""
        # This would use the NER model to extract structured medication information
        # Simplified for demonstration
        return [{'name': 'Sample Medication', 'dosage': '10mg', 'frequency': 'twice daily'}]
    
    def _extract_doctor_info(self, text):
        """Extract doctor information from text"""
        # This would extract doctor name, credentials, etc.
        # Simplified for demonstration
        return {
            'name': 'Dr. Smith',
            'credentials': 'MD',
            'signature': True
        }
    
    def _extract_name(self, text):
        """Extract patient name from text"""
        # Simplified extraction
        return "John Doe"
    
    def _extract_age(self, text):
        """Extract patient age from text"""
        # Simplified extraction
        return "45"
    
    def _extract_gender(self, text):
        """Extract patient gender from text"""
        # Simplified extraction
        return "Male"
    
    def _extract_id(self, text):
        """Extract patient ID from text"""
        # Simplified extraction
        return "12345"
    
    def _extract_date(self, text):
        """Extract date from text"""
        # Simplified extraction
        return "2025-04-27"
    
    def _extract_refills(self, text):
        """Extract refill information from text"""
        # Simplified extraction
        if "refill" in text.lower():
            return "3"
        return None
    
    def validate_extracted_data(self, structured_data):
        """
        Validate and correct extracted data
        
        Args:
            structured_data: Dictionary of extracted structured data
            
        Returns:
            Validated and corrected data with confidence scores
        """
        validated_data = structured_data.copy()
        validated_data['confidence_scores'] = {}
        
        # Validate patient information
        if 'patient' in validated_data:
            self._validate_patient_info(validated_data)
            
        # Validate medication
        if 'medication' in validated_data:
            self._validate_medication_info(validated_data)
            
        # Flag missing required fields
        self._check_missing_required_fields(validated_data)
        
        return validated_data
    
    def _validate_patient_info(self, data):
        """Validate patient information"""
        patient = data['patient']
        confidence = {}
        
        # Check name format
        if 'name' in patient and patient['name']:
            name_parts = patient['name'].split()
            confidence['name'] = 0.9 if len(name_parts) >= 2 else 0.6
        else:
            confidence['name'] = 0.0
            
        # Check age is numeric
        if 'age' in patient and patient['age']:
            try:
                int(patient['age'])
                confidence['age'] = 0.9
            except ValueError:
                confidence['age'] = 0.3
        else:
            confidence['age'] = 0.0
            
        data['confidence_scores']['patient'] = confidence
    
    def _validate_medication_info(self, data):
        """Validate medication information"""
        medications = data['medication']
        med_confidence = []
        
        for i, med in enumerate(medications):
            confidence = {}
            
            # Check drug name against database
            if 'name' in med and med['name']:
                # Fuzzy match against drug database
                best_match, score = self._fuzzy_match_drug(med['name'])
                if score > 0.8:
                    # Correct drug name if good match found
                    med['name'] = best_match
                    confidence['name'] = score
                else:
                    confidence['name'] = score * 0.5
            else:
                confidence['name'] = 0.0
                
            # Check dosage format
            if 'dosage' in med and med['dosage']:
                # Check if dosage contains number + unit
                has_number = any(c.isdigit() for c in med['dosage'])
                has_unit = any(unit in med['dosage'].lower() for unit in self.drug_database['dosage_units'])
                
                if has_number and has_unit:
                    confidence['dosage'] = 0.9
                elif has_number or has_unit:
                    confidence['dosage'] = 0.6
                else:
                    confidence['dosage'] = 0.3
            else:
                confidence['dosage'] = 0.0
                
            # Check frequency against known patterns
            if 'frequency' in med and med['frequency']:
                if any(freq in med['frequency'].lower() for freq in self.drug_database['frequency']):
                    confidence['frequency'] = 0.9
                else:
                    confidence['frequency'] = 0.5
            else:
                confidence['frequency'] = 0.0
                
            med_confidence.append(confidence)
            
        data['confidence_scores']['medication'] = med_confidence
    
    def _fuzzy_match_drug(self, drug_name):
        """Perform fuzzy matching of drug name against database"""
        # In a real implementation, this would use a proper fuzzy matching algorithm
        # Simplified for demonstration
        best_match = ""
        best_score = 0
        
        for drug in self.drug_database['common_drugs']:
            # Simple case-insensitive substring matching
            if drug.lower() in drug_name.lower():
                score = len(drug) / len(drug_name)
                if score > best_score:
                    best_score = score
                    best_match = drug
        
        return best_match if best_match else drug_name, best_score if best_score > 0 else 0.5
    
    def _check_missing_required_fields(self, data):
        """Check for missing required fields"""
        required_fields = {
            'patient': ['name'],
            'medication': ['name', 'dosage', 'frequency'],
            'doctor': ['name']
        }
        
        missing = {}
        
        for section, fields in required_fields.items():
            if section not in data:
                missing[section] = fields
                continue
                
            missing_fields = []
            
            if section == 'medication':
                # Check each medication
                for i, med in enumerate(data[section]):
                    for field in fields:
                        if field not in med or not med[field]:
                            missing_fields.append(f"medication[{i}].{field}")
            else:
                # Check other sections
                for field in fields:
                    if field not in data[section] or not data[section][field]:
                        missing_fields.append(f"{section}.{field}")
            
            if missing_fields:
                missing[section] = missing_fields
        
        if missing:
            data['missing_required_fields'] = missing
    
    def extract_and_process(self, image_path):
        """
        Full pipeline to extract structured data from prescription
        
        Args:
            image_path: Path to prescription image
            
        Returns:
            Structured data extracted from prescription
        """
        # Preprocess image
        preprocessed = self.preprocess_image(image_path)
        
        # Segment prescription
        regions = self.segment_prescription(preprocessed)
        
        # Perform OCR
        extracted_text = self.perform_ocr(regions)
        
        # Extract entities
        structured_data = self.extract_entities(extracted_text)
        
        # Validate data
        validated_data = self.validate_extracted_data(structured_data)
        
        return validated_data

# Training functions for the pipeline components

def build_segmentation_model(input_shape=(512, 512, 1)):
    """Build U-Net style segmentation model"""
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # More encoder layers would go here
    
    # Decoder would go here
    
    # Output layer - 4 regions (background, patient info, medication, doctor info)
    outputs = Conv2D(4, 1, activation='softmax')(conv2)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def build_ocr_model(input_shape=(128, 32, 1), vocab_size=128):
    """Build CRNN model for OCR"""
    inputs = Input(input_shape)
    
    # CNN feature extraction
    x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    
    # Prepare feature map for RNN
    _, h, w, c = x.shape
    x = Reshape((w, h * c))(x)
    
    # RNN for sequence modeling
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    
    # Output layer - one timestep per column, predictions for each character
    outputs = Dense(vocab_size + 1, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    # CTC loss would be used in actual implementation
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    return model

def train_segmentation_model(data_dir, output_model_path, batch_size=8, epochs=10):
    """Train segmentation model"""
    # This would load and prepare dataset, then train the model
    # Simplified for demonstration
    model = build_segmentation_model()
    
    # In a real implementation, we would:
    # 1. Load images and masks
    # 2. Split into train/val
    # 3. Create data generators
    # 4. Train the model
    # 5. Save the trained model
    
    print("Training segmentation model...")
    # model.fit(...) would go here
    
    model.save(output_model_path)
    return model

def train_ocr_model(data_dir, output_model_path, batch_size=32, epochs=20):
    """Train OCR model"""
    # This would load and prepare dataset, then train the model
    # Simplified for demonstration
    model = build_ocr_model()
    
    print("Training OCR model...")
    # model.fit(...) would go here
    
    model.save(output_model_path)
    return model

def train_ner_model(data_dir, output_model_path, batch_size=16, epochs=5):
    """Train NER model"""
    # This would fine-tune a pretrained model on prescription data
    # Simplified for demonstration
    
    print("Training NER model...")
    # In a real implementation, we would use the Transformers library
    # to fine-tune a model for medical NER
    
    return "Trained NER model would be saved here"

# Example of pipeline usage
def demo_pipeline():
    """Demonstrate the prescription extraction pipeline"""
    # Initialize the pipeline
    # In a real implementation, you would provide paths to trained models
    extractor = PrescriptionExtractor()
    
    # Process a sample prescription
    result = extractor.extract_and_process("sample_prescription.jpg")
    
    # Print structured output
    print(json.dumps(result, indent=2))
    
    return result

if __name__ == "__main__":
    # Train models (commented out for demonstration)
    # train_segmentation_model("data/segmentation", "models/segmentation_model.h5")
    # train_ocr_model("data/ocr", "models/ocr_model.h5")
    # train_ner_model("data/ner", "models/ner_model")
    
    # Run demo
    demo_pipeline()
