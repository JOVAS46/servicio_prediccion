import os
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configuración de la base de datos PostgreSQL desde variables de entorno
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'defaultdb'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'sslmode': os.getenv('DB_SSLMODE', 'prefer')
}

# Configuración del modelo
MODEL_FILE = "modelo_consumo.pkl"
MAPPINGS_FILE = "category_mappings.pkl"
BACKUP_DIR = os.getenv('MODEL_BACKUP_DIR', 'model_backups')

# Configuración de Flask
FLASK_CONFIG = {
    'ENV': os.getenv('FLASK_ENV', 'production'),
    'DEBUG': os.getenv('FLASK_DEBUG', 'False').lower() == 'true',
    'PORT': int(os.getenv('FLASK_PORT', 5000))
}

# Configuración de entrenamiento
TRAIN_PARAMS = {
    'n_estimators': int(os.getenv('TRAIN_N_ESTIMATORS', 100)),
    'random_state': int(os.getenv('TRAIN_RANDOM_STATE', 42)),
    'max_depth': int(os.getenv('TRAIN_MAX_DEPTH', 10)),
    'min_samples_split': int(os.getenv('TRAIN_MIN_SAMPLES_SPLIT', 5)),
    'verbose': 1  # Para ver el progreso del entrenamiento
}