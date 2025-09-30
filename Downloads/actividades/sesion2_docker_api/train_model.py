#!/usr/bin/env python3
"""
Script de entrenamiento del modelo para Sesión 2
Entrena RandomForest con dataset Wine de sklearn
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("=== Entrenamiento del Modelo RandomForest con Dataset Wine ===")

    # Cargar dataset Wine
    print("Cargando dataset Wine...")
    wine = load_wine()
    X, y = wine.data, wine.target

    print(f"Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"Clases: {wine.target_names}")

    # Dividir datos para evaluación
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Crear y entrenar modelo RandomForest
    print("Entrenando modelo RandomForest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluar modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Modelo RandomForest entrenado exitosamente.")
    print(f"Precisión en datos de prueba: {accuracy:.2%}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=wine.target_names))

    # Guardar modelo usando joblib
    model_filename = 'modelo.pkl'
    joblib.dump(model, model_filename)
    print(f"\nModelo guardado exitosamente en: '{model_filename}'")

    # Verificar carga del modelo
    print("Verificando carga del modelo...")
    modelo_cargado = joblib.load(model_filename)
    test_prediction = modelo_cargado.predict([X[0]])
    print(f"Predicción de prueba con modelo cargado: {wine.target_names[test_prediction[0]]}")

    return model

if __name__ == "__main__":
    main()