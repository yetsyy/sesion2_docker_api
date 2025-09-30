#!/usr/bin/env python3
"""
API REST con Flask para exponer modelo de clasificación Wine
Sesión 2 - Contenerización de una API ML con Docker
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
import os

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Cargar modelo al iniciar la aplicación
try:
    model = joblib.load('modelo.pkl')
    logger.info("Modelo cargado exitosamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {e}")
    model = None

# Nombres de las clases Wine
class_names = ['class_0', 'class_1', 'class_2']

@app.route('/', methods=['GET'])
def home():
    """Mensaje de bienvenida"""
    return jsonify({
        "message": "API Wine Classifier lista para Docker",
        "description": "API REST para clasificación de vinos usando RandomForest",
        "version": "2.0",
        "docker_ready": True,
        "endpoints": {
            "/": "Mensaje de bienvenida",
            "/predict": "POST - Predicción de clase de vino"
        },
        "example_request": {
            "url": "/predict",
            "method": "POST",
            "body": {
                "features": [1.423e+01, 1.71e+00, 2.43e+00, 1.56e+01, 1.27e+02, 2.8e+00, 3.06e+00, 2.8e-01, 2.29e+00, 5.64e+00, 1.04e+00, 3.92e+00, 1.065e+03]
            }
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para realizar predicciones de clasificación de vinos"""
    try:
        # Verificar que el modelo esté cargado
        if model is None:
            return jsonify({"error": "Modelo no disponible"}), 500

        # Obtener datos JSON del request
        data = request.get_json()

        # Validar que existe la clave 'features'
        if not data or 'features' not in data:
            return jsonify({
                "error": "Formato incorrecto. Se requiere JSON con clave 'features'",
                "example": {"features": [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]}
            }), 400

        features = data['features']

        # Validar que features es una lista
        if not isinstance(features, list):
            return jsonify({
                "error": "El campo 'features' debe ser una lista",
                "received_type": str(type(features).__name__)
            }), 400

        # Validar cantidad de características (Wine tiene 13)
        if len(features) != 13:
            return jsonify({
                "error": f"Se requieren exactamente 13 características. Recibidas: {len(features)}",
                "expected": "13 características del dataset Wine"
            }), 400

        # Validar que todas las características sean numéricas
        try:
            features_array = np.array(features, dtype=float).reshape(1, -1)
        except (ValueError, TypeError):
            return jsonify({
                "error": "Todas las características deben ser valores numéricos",
                "received": features
            }), 400

        # Realizar predicción
        prediction = model.predict(features_array)[0]
        prediction_proba = model.predict_proba(features_array)[0]

        # Formatear respuesta
        response = {
            "prediction": class_names[prediction],
            "prediction_index": int(prediction),
            "confidence": {
                "class_0": float(prediction_proba[0]),
                "class_1": float(prediction_proba[1]),
                "class_2": float(prediction_proba[2])
            },
            "input_features": features,
            "docker_version": True
        }

        logger.info(f"Predicción realizada: {class_names[prediction]} para features de longitud {len(features)}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return jsonify({
            "error": "Error interno del servidor",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for Docker"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": "ready"
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint no encontrado",
        "available_endpoints": ["/", "/predict", "/health"]
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Método no permitido",
        "hint": "Usar GET para '/' y '/health', POST para '/predict'"
    }), 405

if __name__ == '__main__':
    # Verificar que existe el modelo
    if not os.path.exists('modelo.pkl'):
        print("ERROR: No se encontró el archivo 'modelo.pkl'")
        print("Ejecuta primero: python train_model.py")
        exit(1)

    print("Iniciando API Flask para Docker...")
    print("Endpoints disponibles:")
    print("  GET  /        - Mensaje de bienvenida")
    print("  GET  /health  - Health check")
    print("  POST /predict - Predicción de clase Wine")

    # Para Docker, usar host='0.0.0.0' para permitir conexiones externas
    app.run(debug=False, host='0.0.0.0', port=5000)