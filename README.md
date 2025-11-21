# Servicio de IA - Sistema de Predicciones

Servicio Flask para predicciones de machine learning que incluye:
- Predicción de consumo de insumos
- Predicción de desperdicios
- Predicción de ventas e ingresos
- Entrenamiento automático de modelos

## 🚀 Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/JOVAS46/servicio_prediccion.git
cd servicio_prediccion
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**
Copia el archivo `.env.example` a `.env` y configura tus credenciales:
```bash
cp .env.example .env
```

Edita el archivo `.env` con tus datos de PostgreSQL:
```env
DB_HOST=tu_host_postgresql
DB_USER=tu_usuario
DB_PASSWORD=tu_password
DB_NAME=tu_base_datos
DB_PORT=5432
```

5. **Ejecutar el servicio**
```bash
python flask_service_new.py
```

## 📋 Endpoints Disponibles

### 🎯 Predicciones
- `POST /predict/month_insumo` - Predecir consumo total de un insumo por mes
- `POST /predict/month_plato` - Predecir consumo de insumo por receta específica
- `POST /predict/waste` - Predecir desperdicios de un insumo
- `POST /predict/sales` - Predecir ventas e ingresos

### 🤖 Entrenamiento
- `POST /train/recetas` - Entrenar modelo de consumo basado en recetas
- `POST /train/sales` - Entrenar modelo de ventas

### 📊 Información
- `GET /get/insumos_disponibles` - Lista de insumos en la BD
- `GET /get/recetas_disponibles` - Lista de recetas disponibles

## 🐳 Docker

```bash
# Construir imagen
docker build -t servicio-ia .

# Ejecutar contenedor
docker-compose up -d
```

## 🔧 Configuración

El servicio utiliza variables de entorno definidas en `.env`:

- **DB_HOST**: Host de PostgreSQL
- **DB_USER**: Usuario de base de datos
- **DB_PASSWORD**: Contraseña
- **DB_NAME**: Nombre de la base de datos
- **DB_PORT**: Puerto (por defecto 5432)
- **FLASK_PORT**: Puerto del servicio (por defecto 5000)
- **FLASK_DEBUG**: Modo debug (True/False)

## 📁 Estructura

```
servicio-ia/
├── flask_service_new.py    # Servicio principal
├── config.py              # Configuración
├── database.py            # Funciones de BD
├── model_training.py      # Entrenamiento de modelos
├── requirements.txt       # Dependencias
├── .env                   # Variables de entorno
├── Dockerfile            # Containerización
└── docker-compose.yml   # Orquestación
```

## 🧠 Modelos de ML

El servicio entrena automáticamente modelos RandomForest para:
1. **Predicción de consumo**: Basado en ventas históricas y recetas
2. **Predicción de ventas**: Basado en histórico de ventas por producto
3. **Análisis de desperdicios**: Calcula desperdicios por receta e insumo

## 📝 Ejemplos de Uso

### Predecir consumo de un insumo
```bash
curl -X POST http://localhost:5000/predict/month_insumo \
  -H "Content-Type: application/json" \
  -d '{"insumo": "tomate", "mes": "diciembre"}'
```

### Predecir desperdicios
```bash
curl -X POST http://localhost:5000/predict/waste \
  -H "Content-Type: application/json" \
  -d '{"insumo": "lechuga"}'
```

### Entrenar modelo
```bash
curl -X POST http://localhost:5000/train/recetas
```

## 🔒 Seguridad

- Las credenciales se manejan mediante variables de entorno
- Conexiones SSL a PostgreSQL
- Validación de entrada en todos los endpoints