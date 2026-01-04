"""
LRI (Limbal Relaxing Incision) Prediction Module
=================================================
This module provides inference capabilities for predicting:
1. Arcuate type (None/Single/Paired) - Model 1
2. LRI length for single arcuate cases - Model 2
3. LRI length for paired arcuate cases - Model 3

Usage:
    from inference import LRIPredictor
    
    predictor = LRIPredictor()
    result = predictor.predict(patient_data)
"""

import json
import pickle
import numpy as np
import xgboost as xgb
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def get_magnitude_at_axis(magnitude: float, angle_degrees: float, desired_axis_degrees: float) -> float:
    """
    Convert a single astigmatism measurement to its magnitude at a specified axis.
    
    Args:
        magnitude: Original magnitude of astigmatism (in diopters, positive cylinder)
        angle_degrees: Original axis of astigmatism (in degrees)
        desired_axis_degrees: Axis at which to calculate the magnitude (in degrees)
    
    Returns:
        float: Magnitude at the desired axis (in diopters)
    """
    if magnitude == 0:
        return 0.0
    
    angle_rad = np.radians(2 * (desired_axis_degrees - angle_degrees))
    magnitude_at_desired_axis = magnitude * np.cos(angle_rad)
    
    return magnitude_at_desired_axis


def convert_neg_to_pos_cyl(neg_cyl: float, neg_axis: float) -> Tuple[float, float]:
    """
    Convert negative cylinder notation to positive cylinder notation.
    
    Args:
        neg_cyl: Cylinder value in negative format (negative number)
        neg_axis: Axis in negative cylinder notation (degrees)
    
    Returns:
        tuple: (positive_cylinder, converted_axis)
    """
    pos_cyl = -neg_cyl
    converted_axis = neg_axis + 90
    if converted_axis > 180:
        converted_axis -= 180
    
    return pos_cyl, converted_axis


# ============================================================================
# PREDICTION RESULT DATACLASS
# ============================================================================

@dataclass
class LRIPrediction:
    """Result of LRI prediction"""
    arcuate_type: str           # "None", "Single", or "Paired"
    arcuate_code: int           # 0, 1, or 2
    lri_length: Optional[float] # None if arcuate_type is "None", otherwise predicted length
    lri_axis: Optional[int]     # Barrett Integrated-K axis (degrees) - the axis for the arcuate(s)
    num_arcuates: int           # 0, 1, or 2
    recommendation: str         # Human-readable recommendation
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# MAIN PREDICTOR CLASS
# ============================================================================

class LRIPredictor:
    """
    LRI Prediction Engine
    
    Loads trained XGBoost models and provides inference for new patients.
    
    Workflow:
    1. User inputs patient parameters
    2. Parameters are preprocessed (e.g., axis → cos/sin components)
    3. Model 1 classifies: None (0), Single (1), or Paired (2)
    4. Based on classification:
       - None: No arcuate recommended
       - Single: Model 2 predicts LRI length, axis = Barrett Integrated-K axis
       - Paired: Model 3 predicts LRI length (same for both), axis = Barrett Integrated-K axis
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the predictor by loading all models and configs.
        
        Args:
            models_dir: Path to directory containing saved models
        """
        self.models_dir = Path(models_dir)
        self._load_models()
    
    def _load_models(self):
        """Load all models and configurations."""
        # Load configurations
        with open(self.models_dir / "model_configs.json", "r") as f:
            self.configs = json.load(f)
        
        # Load label encoder for Model 1
        with open(self.models_dir / "label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)
        
        # Load XGBoost models
        self.model1 = xgb.Booster()
        self.model1.load_model(str(self.models_dir / "model1_classification.json"))
        
        self.model2 = xgb.Booster()
        self.model2.load_model(str(self.models_dir / "model2_single_lri.json"))
        
        self.model3 = xgb.Booster()
        self.model3.load_model(str(self.models_dir / "model3_paired_lri.json"))
        
        # Store feature columns for each model
        self.model1_features = self.configs['model1']['feature_columns']
        self.model2_features = self.configs['model2']['feature_columns']
        self.model3_features = self.configs['model3']['feature_columns']
        
        # Class mapping for Model 1
        self.class_mapping = {0: "None", 1: "Single", 2: "Paired"}
    
    def preprocess_patient(self, patient_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Preprocess raw patient data into model features.
        
        Args:
            patient_data: Dictionary with raw patient measurements:
                - age: Patient age
                - laterality: "OD" or "OS"
                - manifest_cylinder: Manifest cylinder (negative value)
                - manifest_axis: Manifest axis (degrees)
                - barrett_k_magnitude: Barrett Integrated-K magnitude (D)
                - barrett_k_axis: Barrett Integrated-K axis (degrees)
                - delta_k_iol700_magnitude: ΔK IOL 700 magnitude (D)
                - delta_k_iol700_axis: ΔK IOL 700 axis (degrees)
                - delta_tk_iol700_magnitude: ΔTK IOL 700 magnitude (D)
                - delta_tk_iol700_axis: ΔTK IOL 700 axis (degrees)
                - post_astig_iol700_magnitude: Post. astigmatism IOL 700 magnitude (D)
                - post_astig_iol700_axis: Post. astigmatism IOL 700 axis (degrees)
                - pentacam_delta_k_magnitude: ΔK Pentacam magnitude (D)
                - pentacam_delta_k_axis: ΔK Pentacam axis (degrees)
                - axial_length: Axial length (mm)
        
        Returns:
            Dictionary with all computed features
        """
        bik_axis = patient_data['barrett_k_axis']
        
        # Compute BIK axis angular features (from Barrett Integrated-K axis)
        bik_axis_cos = np.cos(np.radians(bik_axis * 2))
        bik_axis_sin = np.sin(np.radians(bik_axis * 2))
        
        # Convert manifest cylinder to magnitude at BIK axis
        # Only convert if cylinder is in negative notation (negative value)
        if patient_data['manifest_cylinder'] < 0:
            pos_cyl, converted_axis = convert_neg_to_pos_cyl(
                patient_data['manifest_cylinder'],
                patient_data['manifest_axis']
            )
        else:
            # Already in positive notation, use as-is
            pos_cyl = patient_data['manifest_cylinder']
            converted_axis = patient_data['manifest_axis']
        manifest_cyl_at_bik = get_magnitude_at_axis(pos_cyl, converted_axis, bik_axis)
        
        # Compute other features at BIK axis
        delta_k_at_bik = get_magnitude_at_axis(
            patient_data['delta_k_iol700_magnitude'],
            patient_data['delta_k_iol700_axis'],
            bik_axis
        )
        
        delta_tk_at_bik = get_magnitude_at_axis(
            patient_data['delta_tk_iol700_magnitude'],
            patient_data['delta_tk_iol700_axis'],
            bik_axis
        )
        
        post_astig_at_bik = get_magnitude_at_axis(
            patient_data['post_astig_iol700_magnitude'],
            patient_data['post_astig_iol700_axis'],
            bik_axis
        )
        
        pentacam_at_bik = get_magnitude_at_axis(
            patient_data['pentacam_delta_k_magnitude'],
            patient_data['pentacam_delta_k_axis'],
            bik_axis
        )
        
        # Encode laterality (OD=0, OS=1)
        laterality_code = 0 if patient_data['laterality'].upper() == 'OD' else 1
        
        return {
            'Age': patient_data['age'],
            'Laterality': laterality_code,
            'Barrett Integrated-K magnitude (D)': patient_data['barrett_k_magnitude'],
            'BIK_axis_cos': bik_axis_cos,
            'BIK_axis_sin': bik_axis_sin,
            'deltaTK_IOL700_cyl_atBIKaxis': delta_tk_at_bik,
            'Manifest_cyl_at_BIKaxis': manifest_cyl_at_bik,
            'deltaK_IOL700_cyl_atBIKaxis': delta_k_at_bik,
            'PostAstig_IOL700_cyl_atBIKaxis': post_astig_at_bik,
            'Pentacam_cyl_atBIKaxis': pentacam_at_bik,
            'Axial length (mm)': patient_data['axial_length']
        }
    
    def predict_arcuate_type(self, features: Dict[str, float]) -> Tuple[str, int]:
        """
        Predict arcuate type using Model 1.
        
        Args:
            features: Preprocessed feature dictionary
        
        Returns:
            Tuple of (arcuate_type_str, arcuate_code)
        """
        feature_values = [features[col] for col in self.model1_features]
        dmatrix = xgb.DMatrix([feature_values], feature_names=self.model1_features)
        pred_code = int(self.model1.predict(dmatrix)[0])
        pred_type = self.class_mapping[pred_code]
        
        return pred_type, pred_code
    
    def predict_lri_length_single(self, features: Dict[str, float]) -> float:
        """
        Predict LRI length for single arcuate using Model 2.
        """
        feature_values = [features[col] for col in self.model2_features]
        dmatrix = xgb.DMatrix([feature_values], feature_names=self.model2_features)
        return float(self.model2.predict(dmatrix)[0])
    
    def predict_lri_length_paired(self, features: Dict[str, float]) -> float:
        """
        Predict LRI length for paired arcuates using Model 3.
        """
        feature_values = [features[col] for col in self.model3_features]
        dmatrix = xgb.DMatrix([feature_values], feature_names=self.model3_features)
        return float(self.model3.predict(dmatrix)[0])
    
    def predict(self, patient_data: Dict[str, Any]) -> LRIPrediction:
        """
        Full prediction pipeline for AUTO SELECT mode.
        
        Workflow:
        1. Preprocess patient data
        2. Run Model 1 to classify arcuate type
        3. Based on result:
           - None: Return "No arcuates recommended"
           - Single: Run Model 2, return length + axis
           - Paired: Run Model 3, return length + axis
        
        Args:
            patient_data: Raw patient measurements dictionary
        
        Returns:
            LRIPrediction with complete recommendation
        """
        # Store the Barrett Integrated-K axis for output
        bik_axis = int(patient_data['barrett_k_axis'])
        
        # Preprocess patient data
        features = self.preprocess_patient(patient_data)
        
        # Threshold rules based on astigmatism type and magnitude
        # BIK_axis_cos: +1 = WTR (0°/180°), -1 = ATR (90°), 0 = oblique
        bik_axis_cos = features['BIK_axis_cos']
        barrett_k_mag = patient_data['barrett_k_magnitude']
        
        # Apply threshold rules before model prediction
        # In positive cylinder notation: axis 0°/180° = ATR, axis 90° = WTR
        # cos(2*axis): +1 at 0°/180° (ATR), -1 at 90° (WTR)
        if bik_axis_cos > 0.5:
            # ATR astigmatism (axis near 0°/180°): None if Barrett K < 0.1
            if barrett_k_mag < 0.1:
                return LRIPrediction(
                    arcuate_type="None",
                    arcuate_code=0,
                    lri_length=None,
                    lri_axis=None,
                    num_arcuates=0,
                    recommendation="No arcuates recommended (ATR astigmatism below threshold)"
                )
        elif bik_axis_cos < -0.55:
            # WTR astigmatism (axis near 90°): None if Barrett K < 0.3
            if barrett_k_mag < 0.3:
                return LRIPrediction(
                    arcuate_type="None",
                    arcuate_code=0,
                    lri_length=None,
                    lri_axis=None,
                    num_arcuates=0,
                    recommendation="No arcuates recommended (WTR astigmatism below threshold)"
                )
        else:
            # Oblique astigmatism (axis near 45°/135°): None if Barrett K < 0.2
            if barrett_k_mag < 0.2:
                return LRIPrediction(
                    arcuate_type="None",
                    arcuate_code=0,
                    lri_length=None,
                    lri_axis=None,
                    num_arcuates=0,
                    recommendation="No arcuates recommended (oblique astigmatism below threshold)"
                )
        
        # Step 1: Classify arcuate type using Model 1
        arcuate_type, arcuate_code = self.predict_arcuate_type(features)
        
        # Step 2: Based on classification, determine output
        if arcuate_type == "None":
            return LRIPrediction(
                arcuate_type="None",
                arcuate_code=0,
                lri_length=None,
                lri_axis=None,
                num_arcuates=0,
                recommendation="No arcuates recommended"
            )
        
        elif arcuate_type == "Single":
            lri_length = self.predict_lri_length_single(features)
            return LRIPrediction(
                arcuate_type="Single",
                arcuate_code=1,
                lri_length=round(lri_length, 1),
                lri_axis=bik_axis,
                num_arcuates=1,
                recommendation=f"Single arcuate: {round(lri_length, 1)}° length at {bik_axis}° axis"
            )
        
        else:  # Paired
            lri_length = self.predict_lri_length_paired(features)
            return LRIPrediction(
                arcuate_type="Paired",
                arcuate_code=2,
                lri_length=round(lri_length, 1),
                lri_axis=bik_axis,
                num_arcuates=2,
                recommendation=f"Paired arcuates: {round(lri_length, 1)}° length each at {bik_axis}° axis"
            )
    
    def predict_single_only(self, patient_data: Dict[str, Any]) -> LRIPrediction:
        """
        Predict LRI length assuming SINGLE arcuate (skip Model 1).
        Use when user manually selects "Single" mode.
        """
        bik_axis = int(patient_data['barrett_k_axis'])
        features = self.preprocess_patient(patient_data)
        lri_length = self.predict_lri_length_single(features)
        
        return LRIPrediction(
            arcuate_type="Single",
            arcuate_code=1,
            lri_length=round(lri_length, 1),
            lri_axis=bik_axis,
            num_arcuates=1,
            recommendation=f"Single arcuate: {round(lri_length, 1)}° length at {bik_axis}° axis"
        )
    
    def predict_paired_only(self, patient_data: Dict[str, Any]) -> LRIPrediction:
        """
        Predict LRI length assuming PAIRED arcuates (skip Model 1).
        Use when user manually selects "Paired" mode.
        """
        bik_axis = int(patient_data['barrett_k_axis'])
        features = self.preprocess_patient(patient_data)
        lri_length = self.predict_lri_length_paired(features)
        
        return LRIPrediction(
            arcuate_type="Paired",
            arcuate_code=2,
            lri_length=round(lri_length, 1),
            lri_axis=bik_axis,
            num_arcuates=2,
            recommendation=f"Paired arcuates: {round(lri_length, 1)}° length each at {bik_axis}° axis"
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_predictor(models_dir: str = "models") -> LRIPredictor:
    """Create and return a predictor instance."""
    return LRIPredictor(models_dir)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example patient data
    example_patient = {
        'age': 68,
        'laterality': 'OD',
        'manifest_cylinder': -1.50,
        'manifest_axis': 180,
        'barrett_k_magnitude': 1.25,
        'barrett_k_axis': 85,
        'delta_k_iol700_magnitude': 0.45,
        'delta_k_iol700_axis': 88,
        'delta_tk_iol700_magnitude': 0.52,
        'delta_tk_iol700_axis': 92,
        'post_astig_iol700_magnitude': 0.1,
        'post_astig_iol700_axis': 178,
        'pentacam_delta_k_magnitude': 0.41,
        'pentacam_delta_k_axis': 87,
        'axial_length': 23.5
    }
    
    try:
        predictor = LRIPredictor()
        
        print("=" * 60)
        print("LRI PREDICTION - AUTO SELECT MODE")
        print("=" * 60)
        
        result = predictor.predict(example_patient)
        
        print(f"\nRecommendation: {result.recommendation}")
        print(f"\nDetails:")
        print(f"  Arcuate Type: {result.arcuate_type}")
        print(f"  Number of Arcuates: {result.num_arcuates}")
        if result.lri_length is not None:
            print(f"  LRI Length: {result.lri_length}°")
            print(f"  LRI Axis: {result.lri_axis}°")
        
        print("\n" + "=" * 60)
        print("JSON Output:")
        print(json.dumps(result.to_dict(), indent=2))
        
    except FileNotFoundError as e:
        print("Models not found. Please run the notebook first to train and save models.")
        print(f"Error: {e}")
