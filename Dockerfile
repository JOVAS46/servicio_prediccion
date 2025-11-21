# Usar una imagen base de Python
FROM python:3.12

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos del proyecto al contenedor
COPY . /app

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Ejecutar el script de entrenamiento del modelo
RUN python train_model.py

# Exponer el puerto para el servicio Flask
EXPOSE 5000

# Comando por defecto para ejecutar el servicio Flask
CMD ["python", "flask_service.py"]