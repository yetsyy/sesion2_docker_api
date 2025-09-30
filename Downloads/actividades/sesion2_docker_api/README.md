# Sesión 2: Contenerización de una API ML con Docker

API REST con Flask para clasificación de vinos, contenerizada con Docker.

## Archivos incluidos

- `train_model.py`: Script para entrenar el modelo RandomForest con dataset Wine
- `app.py`: API Flask containerizada con endpoints `/`, `/health` y `/predict`
- `modelo.pkl`: Modelo entrenado y serializado (se genera automáticamente)
- `Dockerfile`: Configuración para construir la imagen Docker
- `requirements.txt`: Dependencias Python necesarias
- `docker_commands.txt`: Comandos útiles para Docker
- `README.md`: Esta documentación

## Construcción y Ejecución con Docker

### 1. Construir la imagen Docker
```bash
docker build -t wine-classifier:latest .
```

### 2. Ejecutar el contenedor
```bash
docker run -p 5000:5000 wine-classifier:latest
```

O en modo background:
```bash
docker run -d -p 5000:5000 --name wine-api wine-classifier:latest
```

### 3. Verificar que funciona
```bash
curl http://localhost:5000/
curl http://localhost:5000/health
```

## Endpoints

### GET /
Mensaje de bienvenida y descripción de la API.

### GET /health
Health check endpoint para Docker - verifica que la API y el modelo estén funcionando.

### POST /predict
Realiza predicción de clase de vino basada en 13 características químicas.

**Formato de entrada:**
```json
{
  "features": [alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280_od315_of_diluted_wines, proline]
}
```

**Ejemplo:**
```json
{
  "features": [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]
}
```

**Respuesta:**
```json
{
  "prediction": "class_0",
  "prediction_index": 0,
  "confidence": {
    "class_0": 0.95,
    "class_1": 0.03,
    "class_2": 0.02
  },
  "input_features": [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065],
  "docker_version": true
}
```

## Ejemplos de prueba con curl

```bash
# Predicción clase 0 (ejemplo del dataset Wine)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]}'

# Predicción clase 1
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [12.37, 0.94, 1.36, 10.6, 88, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82, 520]}'

# Predicción clase 2
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [12.86, 1.35, 2.32, 18, 122, 1.51, 1.25, 0.21, 0.94, 4.1, 0.76, 1.29, 630]}'
```

## Características del modelo

- **Dataset**: Wine (178 muestras, 13 características)
- **Algoritmo**: RandomForestClassifier (100 árboles)
- **Clases**: class_0, class_1, class_2
- **Características**: 13 medidas químicas de vinos

## Comandos Docker útiles

Ver logs del contenedor:
```bash
docker logs wine-api
```

Parar el contenedor:
```bash
docker stop wine-api
```

Remover el contenedor:
```bash
docker rm wine-api
```

Entrar al contenedor (debug):
```bash
docker exec -it wine-api /bin/bash
```

## Desarrollo local (sin Docker)

Si prefieres ejecutar sin Docker:

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Entrenar modelo:
```bash
python train_model.py
```

3. Ejecutar API:
```bash
python app.py
```

## Validación y manejo de errores

La API incluye validación completa:
- Verificación del formato JSON
- Validación del número de características (debe ser 13)
- Verificación de tipos numéricos
- Health check endpoint
- Logging para debugging

Los errores devuelven códigos HTTP apropiados (400 para errores de cliente, 500 para errores del servidor).