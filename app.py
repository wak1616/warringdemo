"""
LRI Prediction API
==================
Flask backend for LRI (Limbal Relaxing Incision) predictions.
Designed for deployment on Render.

Endpoints:
    POST /predict_warring          - Auto-select mode (runs Model 1 first)
    POST /predict_warring/single   - Single arcuate only (skip Model 1)
    POST /predict_warring/paired   - Paired arcuates only (skip Model 1)
    GET  /health_warring           - Health check
    GET  /api/info_warring         - API documentation
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import LRIPredictor

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes (needed for Vercel frontend)
CORS(app, origins=["*"])  # In production, replace "*" with your Vercel domain

# Initialize predictor (loads models once at startup)
predictor = None


def get_predictor():
    """Lazy load the predictor."""
    global predictor
    if predictor is None:
        predictor = LRIPredictor(models_dir="models")
    return predictor


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health_warring', methods=['GET'])
def health_check():
    """Health check endpoint for Render."""
    return jsonify({
        'status': 'healthy',
        'message': 'LRI Prediction API is running'
    })


@app.route('/predict_warring', methods=['POST'])
def predict_auto():
    """
    AUTO SELECT mode - runs Model 1 first to determine arcuate type,
    then runs appropriate regression model.
    
    Request body (JSON):
    {
        "age": 68,
        "laterality": "OD",
        "manifest_cylinder": -1.50,
        "manifest_axis": 90,
        "barrett_k_magnitude": 1.25,
        "barrett_k_axis": 85,
        "delta_k_iol700_magnitude": 0.45,
        "delta_k_iol700_axis": 88,
        "delta_tk_iol700_magnitude": 0.52,
        "delta_tk_iol700_axis": 92,
        "post_astig_iol700_magnitude": 0.38,
        "post_astig_iol700_axis": 95,
        "pentacam_delta_k_magnitude": 0.41,
        "pentacam_delta_k_axis": 87,
        "axial_length": 23.5
    }
    
    Response:
    {
        "arcuate_type": "Single",
        "arcuate_code": 1,
        "lri_length": 28.5,
        "lri_axis": 85,
        "num_arcuates": 1,
        "recommendation": "Single arcuate: 28.5° length at 85° axis"
    }
    """
    try:
        patient_data = request.get_json()
        
        if not patient_data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = [
            'age', 'laterality', 'manifest_cylinder', 'manifest_axis',
            'barrett_k_magnitude', 'barrett_k_axis',
            'delta_k_iol700_magnitude', 'delta_k_iol700_axis',
            'delta_tk_iol700_magnitude', 'delta_tk_iol700_axis',
            'post_astig_iol700_magnitude', 'post_astig_iol700_axis',
            'pentacam_delta_k_magnitude', 'pentacam_delta_k_axis',
            'axial_length'
        ]
        
        missing_fields = [f for f in required_fields if f not in patient_data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Run prediction
        pred = get_predictor()
        result = pred.predict(patient_data)
        
        return jsonify(result.to_dict())
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_warring/single', methods=['POST'])
def predict_single():
    """
    SINGLE mode - skip Model 1, directly predict single arcuate length.
    Use when user manually selects "Single" in the frontend.
    """
    try:
        patient_data = request.get_json()
        
        if not patient_data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        pred = get_predictor()
        result = pred.predict_single_only(patient_data)
        
        return jsonify(result.to_dict())
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_warring/paired', methods=['POST'])
def predict_paired():
    """
    PAIRED mode - skip Model 1, directly predict paired arcuate length.
    Use when user manually selects "Paired" in the frontend.
    """
    try:
        patient_data = request.get_json()
        
        if not patient_data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        pred = get_predictor()
        result = pred.predict_paired_only(patient_data)
        
        return jsonify(result.to_dict())
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/info_warring', methods=['GET'])
def api_info():
    """Return API information and expected input format."""
    return jsonify({
        'name': 'LRI Prediction API',
        'version': '1.0.0',
        'endpoints': {
            'POST /predict_warring': 'Auto-select mode (Model 1 classifies, then Model 2 or 3)',
            'POST /predict_warring/single': 'Single arcuate mode (Model 2 only)',
            'POST /predict_warring/paired': 'Paired arcuates mode (Model 3 only)',
            'GET /health_warring': 'Health check',
            'GET /api/info_warring': 'This endpoint'
        },
        'input_fields': {
            'age': 'Patient age (integer)',
            'laterality': 'OD or OS',
            'manifest_cylinder': 'Manifest cylinder in negative notation (e.g., -1.50)',
            'manifest_axis': 'Manifest axis in degrees (1-180)',
            'barrett_k_magnitude': 'Barrett Integrated-K magnitude in diopters',
            'barrett_k_axis': 'Barrett Integrated-K axis in degrees (1-180)',
            'delta_k_iol700_magnitude': 'ΔK IOL 700 magnitude in diopters',
            'delta_k_iol700_axis': 'ΔK IOL 700 axis in degrees',
            'delta_tk_iol700_magnitude': 'ΔTK IOL 700 magnitude in diopters',
            'delta_tk_iol700_axis': 'ΔTK IOL 700 axis in degrees',
            'post_astig_iol700_magnitude': 'Posterior astigmatism IOL 700 magnitude in diopters',
            'post_astig_iol700_axis': 'Posterior astigmatism IOL 700 axis in degrees',
            'pentacam_delta_k_magnitude': 'ΔK Pentacam magnitude in diopters',
            'pentacam_delta_k_axis': 'ΔK Pentacam axis in degrees',
            'axial_length': 'Axial length in mm'
        }
    })


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)
