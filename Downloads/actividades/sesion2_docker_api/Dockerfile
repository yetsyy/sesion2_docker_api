# Usar imagen base oficial de Python 3.9
FROM python:3.9-slim

# Establecer directorio de trabajo en el contenedor
WORKDIR /app

# Copiar archivo de requerimientos
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . .

# Crear el modelo si no existe (para development)
RUN python train_model.py

# Exponer puerto 5000
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]