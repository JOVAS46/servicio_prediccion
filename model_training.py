import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from database import get_training_data, get_categories
from config import MODEL_FILE, MAPPINGS_FILE, TRAIN_PARAMS, BACKUP_DIR

def backup_model():
    """Crea una copia de respaldo del modelo actual"""
    if os.path.exists(MODEL_FILE):
        if not os.path.exists(BACKUP_DIR):
            os.makedirs(BACKUP_DIR)
        backup_path = os.path.join(
            BACKUP_DIR, 
            f"modelo_consumo_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        os.rename(MODEL_FILE, backup_path)
        return True
    return False

def train_model():
    """Entrena el modelo con datos de la base de datos"""
    try:
        # Obtener datos de entrenamiento
        df = get_training_data()
        if df is None or df.empty:
            return False, "No hay datos disponibles para entrenamiento"

        # Obtener categorías actualizadas
        categories = get_categories()
        if categories is None:
            return False, "Error al obtener categorías"

        # Preparar datos para entrenamiento
        X = []
        y = []
        
        for _, row in df.iterrows():
            insumo = categories["INSUMO"].get(row["insumo"].lower(), -1)
            plato = categories["PLATO"].get(row["plato"].lower(), -1)
            if insumo != -1 and plato != -1:
                X.append([
                    insumo,
                    row["dia"],
                    row["mes"],
                    plato,
                    row["desperdicio"]
                ])
                y.append(row["cantidad"])

        if not X or not y:
            return False, "No hay suficientes datos válidos para entrenamiento"

        # Crear y entrenar nuevo modelo
        model = RandomForestRegressor(**TRAIN_PARAMS)
        model.fit(X, y)

        # Hacer backup del modelo actual
        backup_model()

        # Guardar nuevo modelo y mapeos
        joblib.dump(model, MODEL_FILE)
        joblib.dump(categories, MAPPINGS_FILE)

        return True, "Modelo entrenado exitosamente"

    except Exception as e:
        return False, f"Error durante el entrenamiento: {str(e)}"