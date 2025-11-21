import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from datetime import datetime
from config import db_config

def get_db_connection():
    """Establece conexión con la base de datos PostgreSQL"""
    try:
        conn = psycopg2.connect(**db_config)
        return conn
    except psycopg2.Error as err:
        print(f"Error de conexión a la base de datos: {err}")
        return None

def get_training_data():
    """Obtiene los datos de entrenamiento de la base de datos"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        query = """
            SELECT 
                i.nombre as insumo,
                r.nombre as receta,
                EXTRACT(DOW FROM v.created_at)::integer + 1 as dia,
                EXTRACT(MONTH FROM v.created_at)::integer as mes,
                (ri.cantidad * v.cantidad) as cantidad_insumo,
                COALESCE(ri.desperdicio, 5) as desperdicio
            FROM ventas v
            INNER JOIN recetas r ON v.receta_id = r.id
            INNER JOIN recetas_insumos ri ON r.id = ri.receta_id
            INNER JOIN insumos i ON ri.insumo_id = i.id
            WHERE v.created_at IS NOT NULL
                AND v.cantidad > 0
                AND ri.cantidad > 0
            ORDER BY v.created_at DESC
        """
        
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error al obtener datos de entrenamiento: {e}")
        return None
    finally:
        conn.close()

def get_categories():
    """Obtiene las categorías actuales de la base de datos"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        categories = {}
        
        # Obtener insumos
        cursor.execute("SELECT id, nombre FROM insumos ORDER BY id")
        insumos_result = cursor.fetchall()
        categories['INSUMO'] = {row['nombre'].lower().strip(): idx for idx, row in enumerate(insumos_result)}
        
        # Obtener recetas
        cursor.execute("SELECT id, nombre FROM recetas ORDER BY id")
        recetas_result = cursor.fetchall()
        categories['PLATO'] = {row['nombre'].lower().strip(): idx for idx, row in enumerate(recetas_result)}
        
        # Mapeo estático de meses
        categories['MES'] = {
            "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
            "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
            "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
        }
        
        return categories
    except Exception as e:
        print(f"Error al obtener categorías: {e}")
        return None
    finally:
        if conn:
            conn.close()