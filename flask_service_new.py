from flask import Flask, request, jsonify
import joblib
import numpy as np
import subprocess
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from datetime import datetime
import os
import traceback
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from config import db_config, FLASK_CONFIG

app = Flask(__name__)

# Variable global para almacenar el mapeo de nombres de tablas
table_mapping = {}

# Inicializar variables globales
model = None
sales_model = None
category_mappings = {}


def get_db_connection():
    try:
        conn = psycopg2.connect(**db_config)
        return conn
    except psycopg2.Error as err:
        print(f"Error de conexión a la base de datos: {err}")
        return None

# Función para obtener datos de entrenamiento de la BD
def get_training_data():
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        query = """
            SELECT 
                i.nombre as insumo,
                p.nombre as plato,
                DAYOFWEEK(d.fecha) as dia,
                MONTH(d.fecha) as mes,
                d.cantidad,
                COALESCE(d.desperdicio, 0) as desperdicio
            FROM detalle_consumo d
            JOIN insumos i ON d.insumo_id = i.id
            JOIN platos p ON d.plato_id = p.id
            WHERE d.fecha IS NOT NULL
        """
        
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error al obtener datos de entrenamiento: {e}")
        return None
    finally:
        conn.close()

def get_sales_training_data():
    """
    Obtiene datos de ventas desde la tabla 'ventas' y 'recetas'
    ADAPTADO A TU BASE DE DATOS:
    - El precio está en ventas.precio (NO en recetas.precio)
    """
    conn = get_db_connection()
    if not conn:
        print("❌ Error: No se pudo establecer conexión con la base de datos")
        return None
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        print("\n" + "="*60)
        print("🔍 OBTENIENDO DATOS DE VENTAS PARA ENTRENAMIENTO")
        print("="*60)
        
        # 1. Verificar ventas
        cursor.execute("SELECT COUNT(*) as total FROM ventas WHERE cantidad > 0")
        total_ventas = cursor.fetchone()['total']
        print(f"\n✅ Total de ventas válidas: {total_ventas}")
        
        if total_ventas == 0:
            print("❌ No hay ventas con cantidad > 0")
            return None
        
        # 2. Ver muestra de datos
        print("\n📊 Muestra de 5 ventas:")
        cursor.execute("""
            SELECT v.id, v.cantidad, v.precio, v.total, v.created_at, r.nombre
            FROM ventas v
            LEFT JOIN recetas r ON v.receta_id = r.id
            WHERE v.cantidad > 0
            ORDER BY v.created_at DESC
            LIMIT 5
        """)
        muestra = cursor.fetchall()
        for row in muestra:
            print(f"   ID: {row['id']}, Cant: {row['cantidad']}, Precio: {row['precio']}, Total: {row['total']}, Receta: {row['nombre']}")
        
        # 3. CONSULTA PRINCIPAL - Precio está en ventas.precio
        print("\n🔄 Ejecutando consulta principal...")
        query = """
            SELECT 
                r.nombre as producto,
                EXTRACT(MONTH FROM v.created_at)::integer as mes,
                EXTRACT(YEAR FROM v.created_at)::integer as año,
                SUM(v.cantidad) as ventas,
                AVG(v.precio) as precio_venta
            FROM ventas v
            INNER JOIN recetas r ON v.receta_id = r.id
            WHERE v.created_at IS NOT NULL
                AND v.cantidad > 0
                AND r.nombre IS NOT NULL
                AND v.precio > 0
            GROUP BY r.nombre, EXTRACT(MONTH FROM v.created_at), EXTRACT(YEAR FROM v.created_at)
            HAVING SUM(v.cantidad) > 0
            ORDER BY año DESC, mes DESC
        """
        
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print("\n❌ ERROR: La consulta devolvió 0 registros")
            print("\n🔍 Diagnóstico:")
            
            # Ver si hay ventas sin receta
            cursor.execute("""
                SELECT COUNT(*) as total
                FROM ventas v 
                LEFT JOIN recetas r ON v.receta_id = r.id 
                WHERE v.cantidad > 0 AND r.id IS NULL
            """)
            ventas_sin_receta = cursor.fetchone()['total']
            print(f"   Ventas sin receta: {ventas_sin_receta}")
            
            # Ver si hay ventas con precio 0
            cursor.execute("SELECT COUNT(*) as total FROM ventas WHERE cantidad > 0 AND (precio IS NULL OR precio = 0)")
            ventas_sin_precio = cursor.fetchone()['total']
            print(f"   Ventas sin precio: {ventas_sin_precio}")
            
            return None
        
        print(f"\n✅ ÉXITO: {len(df)} registros obtenidos")
        print(f"✅ Productos únicos: {df['producto'].nunique()}")
        print(f"✅ Columnas: {df.columns.tolist()}")
        
        print(f"\n💰 Estadísticas de precios:")
        print(f"   Mínimo: {df['precio_venta'].min():.2f}")
        print(f"   Máximo: {df['precio_venta'].max():.2f}")
        print(f"   Promedio: {df['precio_venta'].mean():.2f}")
        
        print("\n📊 Primeros 5 registros:")
        print(df.head().to_string(index=False))
        
        print("\n📊 Resumen por producto:")
        resumen = df.groupby('producto').agg({
            'ventas': ['sum', 'count', 'mean'],
            'precio_venta': 'mean'
        }).round(2)
        resumen.columns = ['Total_Ventas', 'Registros', 'Promedio_Ventas', 'Precio_Promedio']
        print(resumen)
        
        return df
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        if conn and not conn.closed:
            conn.close()
            print("\n✅ Conexión cerrada")
def update_category_mappings():
    """
    Actualizar mapeos de insumos y recetas desde la base de datos
    """
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        print("🔄 Actualizando mapeos desde la base de datos...")
        
        # Actualizar mapeo de insumos
        cursor.execute("SELECT id, nombre FROM insumos ORDER BY id")
        insumos_result = cursor.fetchall()
        insumos = {row['nombre'].lower().strip(): idx for idx, row in enumerate(insumos_result)}
        
        print(f"   Insumos encontrados: {len(insumos)}")
        for nombre in sorted(insumos.keys())[:5]:  # Mostrar primeros 5
            print(f"     - {nombre}")
        
        # Actualizar mapeo de platos/productos
        cursor.execute("SELECT id, nombre FROM recetas ORDER BY id")
        recetas_result = cursor.fetchall()
        productos = {row['nombre'].lower().strip(): idx for idx, row in enumerate(recetas_result)}
        
        print(f"   Recetas encontradas: {len(productos)}")
        for nombre in sorted(productos.keys())[:5]:  # Mostrar primeros 5
            print(f"     - {nombre}")
        
        # Mapeo de meses (estático)
        meses = {
            "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
            "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
            "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
        }
        
        # Actualizar category_mappings global
        global category_mappings
        category_mappings.update({
            "INSUMO": insumos,
            "PRODUCTO": productos,
            "MES": meses
        })
        
        # Guardar los nuevos mapeos
        try:
            joblib.dump(category_mappings, "category_mappings.pkl")
            print("✅ Mapeos guardados en category_mappings.pkl")
        except Exception as e:
            print(f"⚠️ No se pudieron guardar mapeos: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error al actualizar mapeos: {e}")
        return False
    finally:
        conn.close()

# Función para entrenar el modelo con datos de la BD
def train_model_from_db():
    try:
        # Obtener datos de entrenamiento
        df = get_training_data()
        if df is None or df.empty:
            return False, "No hay datos disponibles para entrenamiento"

        # Actualizar mapeos
        if not update_category_mappings():
            return False, "Error al actualizar mapeos"

        # Preparar datos para entrenamiento
        X = []
        y = []
        
        for _, row in df.iterrows():
            insumo = category_mappings["INSUMO"].get(row["insumo"].lower(), -1)
            plato = category_mappings["PRODUCTO"].get(row["plato"].lower(), -1)
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

        # Entrenar nuevo modelo
        new_model = RandomForestRegressor(n_estimators=100, random_state=42)
        new_model.fit(X, y)

        # Guardar el modelo anterior como respaldo
        if os.path.exists("modelo_consumo.pkl"):
            backup_path = f"modelo_consumo_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            os.rename("modelo_consumo.pkl", backup_path)

        # Guardar el nuevo modelo
        joblib.dump(new_model, "modelo_consumo.pkl")
        global model
        model = new_model

        # También entrenar el modelo de ventas
        df_sales = get_sales_training_data()
        if df_sales is not None and not df_sales.empty:
            le_productos = LabelEncoder()
            df_sales['producto_encoded'] = le_productos.fit_transform(df_sales['producto'])
            
            X_sales = df_sales[['producto_encoded', 'mes', 'año']].values
            y_sales = df_sales['ventas'].values
            
            new_sales_model = RandomForestRegressor(n_estimators=100, random_state=42)
            new_sales_model.fit(X_sales, y_sales)
            
            # Guardar el modelo de ventas
            joblib.dump(new_sales_model, "modelo_ventas.pkl")
            joblib.dump(le_productos, "label_encoder_productos.pkl")
            global sales_model
            sales_model = new_sales_model

        return True, "Modelos entrenados exitosamente"

    except Exception as e:
        return False, f"Error durante el entrenamiento: {str(e)}"


# Endpoint para reentrenar el modelo con datos de la BD
@app.route('/retrain/from_db', methods=['POST'])
def retrain_from_db():
    try:
        success, message = train_model_from_db()
        if success:
            return jsonify({"message": message}), 200
        else:
            return jsonify({"error": message}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/update_mappings', methods=['POST'])
def update_mappings_endpoint():
    try:
        if update_category_mappings():
            return jsonify({"message": "Mapeos actualizados exitosamente"}), 200
        else:
            return jsonify({"error": "Error al actualizar los mapeos"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/predict/waste', methods=['POST'])
def predict_waste():
    """
    Predecir desperdicio diario y semanal de un insumo
    Entrada: {"insumo": "Lechuga"}
    """
    try:
        data = request.get_json()
        
        print("\n" + "="*60)
        print("🗑️ PREDICCIÓN DE DESPERDICIOS")
        print("="*60)
        
        # Validar campo requerido
        if 'insumo' not in data:
            return jsonify({
                "error": "Campo requerido: insumo",
                "ejemplo": {"insumo": "Lechuga"}
            }), 400

        insumo_nombre = data['insumo'].lower().strip()
        print(f"📥 Insumo solicitado: '{insumo_nombre}'")

        # Conectar a la base de datos
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Error de conexión a la base de datos"}), 500

        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # 1. Verificar que el insumo existe y obtener su información
            cursor.execute("""
                SELECT i.id, i.nombre, COALESCE(um.abreviatura, um.nombre, 'unidades') as unidad
                FROM insumos i
                LEFT JOIN unidad_medidas um ON i.unidad_medida_id = um.id
                WHERE LOWER(i.nombre) = %s
            """, (insumo_nombre,))
            
            insumo_info = cursor.fetchone()
            
            if not insumo_info:
                # Obtener lista de insumos disponibles para sugerencia
                cursor.execute("SELECT nombre FROM insumos ORDER BY nombre LIMIT 10")
                insumos_disponibles = [row['nombre'] for row in cursor.fetchall()]
                
                return jsonify({
                    "error": f"Insumo '{data['insumo']}' no encontrado en la base de datos",
                    "insumos_disponibles": insumos_disponibles,
                    "sugerencia": "Verifica que el nombre del insumo sea correcto"
                }), 404

            insumo_id = insumo_info['id']
            unidad = insumo_info['unidad']
            
            print(f"✅ Insumo encontrado: ID={insumo_id}, Nombre='{insumo_info['nombre']}', Unidad={unidad}")

            # 2. Obtener recetas que usan este insumo y calcular desperdicios
            cursor.execute("""
                SELECT 
                    r.id as receta_id,
                    r.nombre as receta,
                    ri.cantidad as cantidad_por_receta,
                    COALESCE(ri.desperdicio, 5) as porcentaje_desperdicio,
                    COUNT(v.id) as total_ventas_historicas,
                    COALESCE(SUM(v.cantidad), 0) as total_vendido,
                    COALESCE(AVG(v.cantidad), 0) as promedio_ventas_diarias
                FROM recetas r
                INNER JOIN recetas_insumos ri ON r.id = ri.receta_id
                LEFT JOIN ventas v ON r.id = v.receta_id 
                    AND v.created_at >= CURRENT_TIMESTAMP - INTERVAL '90 days'
                    AND v.cantidad > 0
                WHERE ri.insumo_id = %s
                    AND ri.cantidad > 0
                GROUP BY r.id, r.nombre, ri.cantidad, ri.desperdicio
                ORDER BY total_vendido DESC
            """, (insumo_id,))
            
            recetas_info = cursor.fetchall()
            
            if not recetas_info:
                return jsonify({
                    "error": f"El insumo '{data['insumo']}' no está asignado a ninguna receta",
                    "sugerencia": "Verifica que el insumo esté configurado en al menos una receta con cantidad > 0"
                }), 404

            print(f"✅ Recetas que usan el insumo: {len(recetas_info)}")

            # 3. Calcular desperdicios por receta
            total_desperdicio_diario = 0
            recetas_con_desperdicio = []
            total_ventas_consideradas = 0
            
            for receta in recetas_info:
                # Cantidad de insumo usada por cada venta de la receta
                cantidad_insumo_por_venta = float(receta['cantidad_por_receta'])
                porcentaje_desp = float(receta['porcentaje_desperdicio'])
                
                # Si hay ventas históricas, usar promedio, sino usar 1 venta estimada por día
                if receta['total_ventas_historicas'] > 0:
                    ventas_diarias_estimadas = max(float(receta['promedio_ventas_diarias']), 0.1)
                else:
                    # Si no hay ventas históricas, asumir 1 venta cada 3 días
                    ventas_diarias_estimadas = 0.33
                
                # Calcular desperdicio diario de esta receta
                consumo_diario = cantidad_insumo_por_venta * ventas_diarias_estimadas
                desperdicio_diario_receta = consumo_diario * (porcentaje_desp / 100)
                
                total_desperdicio_diario += desperdicio_diario_receta
                total_ventas_consideradas += receta['total_ventas_historicas']
                
                recetas_con_desperdicio.append({
                    "receta": receta['receta'],
                    "cantidad_por_receta": round(cantidad_insumo_por_venta, 2),
                    "porcentaje_desperdicio": round(porcentaje_desp, 1),
                    "ventas_diarias_estimadas": round(ventas_diarias_estimadas, 2),
                    "desperdicio_diario": round(desperdicio_diario_receta, 2),
                    "total_ventas_historicas": receta['total_ventas_historicas']
                })
                
                print(f"   🍽️ {receta['receta']}: {consumo_diario:.2f} {unidad}/día → {desperdicio_diario_receta:.2f} {unidad} desperdicio")

            # 4. Calcular totales y estadísticas
            desperdicio_semanal = total_desperdicio_diario * 7
            desperdicio_mensual = total_desperdicio_diario * 30
            
            # Porcentaje promedio ponderado por cantidad de insumo usada
            porcentaje_promedio = sum(
                r['porcentaje_desperdicio'] * r['cantidad_por_receta'] 
                for r in recetas_con_desperdicio
            ) / sum(r['cantidad_por_receta'] for r in recetas_con_desperdicio) if recetas_con_desperdicio else 0
            
            # Ordenar recetas por desperdicio descendente
            recetas_con_desperdicio.sort(key=lambda x: x['desperdicio_diario'], reverse=True)
            
            # 5. Determinar calidad de la predicción
            if total_ventas_consideradas >= 30:
                calidad_prediccion = "Alta - Basada en ventas históricas"
                confianza = "alta"
            elif total_ventas_consideradas >= 10:
                calidad_prediccion = "Media - Algunas ventas históricas"
                confianza = "media"
            else:
                calidad_prediccion = "Baja - Basada en estimaciones"
                confianza = "baja"
            
            # 6. Determinar tendencia simple
            if len(recetas_con_desperdicio) > 1:
                receta_principal = recetas_con_desperdicio[0]
                if receta_principal['total_ventas_historicas'] > 5:
                    tendencia = "creciente" if receta_principal['ventas_diarias_estimadas'] > 0.5 else "estable"
                else:
                    tendencia = "estable"
            else:
                tendencia = "estable"

            print(f"\n📊 RESULTADOS CALCULADOS:")
            print(f"   💧 Desperdicio diario: {total_desperdicio_diario:.2f} {unidad}")
            print(f"   📅 Desperdicio semanal: {desperdicio_semanal:.2f} {unidad}")
            print(f"   📅 Desperdicio mensual: {desperdicio_mensual:.2f} {unidad}")
            print(f"   📊 Porcentaje promedio: {porcentaje_promedio:.1f}%")
            print(f"   🎯 Confianza: {confianza} ({total_ventas_consideradas} ventas analizadas)")
            print(f"   📈 Tendencia: {tendencia}")

            return jsonify({
                "insumo": insumo_info['nombre'],
                "unidad": unidad,
                "prediccion": {
                    "desperdicio_diario": round(total_desperdicio_diario, 2),
                    "desperdicio_semanal": round(desperdicio_semanal, 2),
                    "desperdicio_mensual": round(desperdicio_mensual, 2),
                    "porcentaje_desperdicio_promedio": round(porcentaje_promedio, 1)
                },
                "estadisticas_historicas": {
                    "total_desperdiciado": round(desperdicio_mensual, 2),
                    "dias_analizados": 30,  # Estimación mensual
                    "basado_en": "analisis_recetas_y_ventas",
                    "recetas_analizadas": len(recetas_con_desperdicio),
                    "total_ventas_historicas": total_ventas_consideradas,
                    "calidad_prediccion": calidad_prediccion,
                    "confianza": confianza
                },
                "tendencia": {
                    "direccion": tendencia,
                    "porcentaje_cambio": 0,
                    "descripcion": f"Predicción basada en {len(recetas_con_desperdicio)} receta(s) que usan {insumo_info['nombre']}"
                },
                "recetas_mas_desperdicio": [
                    {
                        "receta": r['receta'],
                        "desperdicio_total": r['desperdicio_diario'],
                        "porcentaje_promedio": r['porcentaje_desperdicio'],
                        "unidad": unidad,
                        "ventas_estimadas": r['ventas_diarias_estimadas'],
                        "cantidad_por_receta": r['cantidad_por_receta']
                    }
                    for r in recetas_con_desperdicio[:5]
                ],
                "analisis_detallado": {
                    "metodo_calculo": "consumo_por_receta_x_ventas_estimadas_x_porcentaje_desperdicio",
                    "factores_considerados": [
                        "Cantidad de insumo por receta",
                        "Porcentaje de desperdicio configurado",
                        "Histórico de ventas por receta",
                        "Estimación de ventas futuras"
                    ]
                },
                "recomendaciones": {
                    "accion_principal": f"Revisar las recetas con mayor desperdicio de {insumo_info['nombre']}",
                    "receta_critica": recetas_con_desperdicio[0]['receta'] if recetas_con_desperdicio else None,
                    "ahorro_potencial": f"Reducir 1% el desperdicio ahorraría {round(total_desperdicio_diario * 0.01 * 30, 2)} {unidad}/mes",
                    "acciones_sugeridas": [
                        "Optimizar cantidad de ingredientes por receta",
                        "Mejorar técnicas de preparación",
                        "Controlar mejor el almacenamiento",
                        "Capacitar al personal en reducción de desperdicios"
                    ]
                }
            }), 200

        finally:
            if conn and not conn.closed:
                conn.close()

    except Exception as e:
        print(f"\n❌ Error en predicción de desperdicios: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Error al calcular desperdicios: {str(e)}",
            "tipo_error": "error_interno",
            "sugerencia": "Verifica que el insumo exista y esté configurado en recetas"
        }), 500

@app.route('/train/recetas', methods=['POST'])
@app.route('/train/recetas', methods=['POST'])
def train_recetas():
    """
    Entrenar modelo de consumo de insumos basado SOLO en:
    - ventas (cantidad vendida, fecha)
    - recetas (qué se vendió)
    - recetas_insumos (cuánto insumo usa cada receta)
    
    NO usa movimiento_inventarios
    """
    try:
        print("\n" + "="*60)
        print("🚀 ENTRENAMIENTO DE MODELO DE CONSUMO POR RECETAS")
        print("="*60)
        
        # Conectar a la base de datos
        conn = get_db_connection()
        if not conn:
            return jsonify({
                "error": "No se pudo conectar a la base de datos",
                "sugerencia": "Verifica las credenciales en db_config"
            }), 500

        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # 1. Verificar datos disponibles
            print("\n🔍 Verificando datos disponibles...")
            
            cursor.execute("SELECT COUNT(*) as total FROM ventas")
            total_ventas = cursor.fetchone()['total']
            print(f"   Ventas registradas: {total_ventas}")
            
            cursor.execute("SELECT COUNT(*) as total FROM recetas")
            total_recetas = cursor.fetchone()['total']
            print(f"   Recetas disponibles: {total_recetas}")
            
            cursor.execute("SELECT COUNT(*) as total FROM recetas_insumos")
            total_recetas_insumos = cursor.fetchone()['total']
            print(f"   Relaciones receta-insumo: {total_recetas_insumos}")
            
            if total_ventas == 0:
                return jsonify({
                    "error": "No hay ventas registradas",
                    "sugerencia": "Necesitas registrar ventas de recetas"
                }), 400
            
            if total_recetas == 0:
                return jsonify({
                    "error": "No hay recetas en la base de datos",
                    "sugerencia": "Crea recetas primero"
                }), 400
            
            if total_recetas_insumos == 0:
                return jsonify({
                    "error": "No hay insumos asignados a las recetas",
                    "sugerencia": "Asigna insumos a tus recetas con cantidades"
                }), 400

            # 2. Consulta principal: calcular consumo real de insumos por venta
            print("\n📊 Obteniendo datos de entrenamiento...")
            print("   Calculando: ventas × recetas × insumos...")
            
            query = """
                SELECT 
                    i.nombre as insumo,
                    r.nombre as plato,
                    EXTRACT(DOW FROM v.created_at)::integer + 1 as dia,
                    EXTRACT(MONTH FROM v.created_at)::integer as mes,
                    (ri.cantidad * v.cantidad) as cantidad_consumida,
                    COALESCE(ri.desperdicio / 100, 0.05) as desperdicio
                FROM ventas v
                INNER JOIN recetas r ON v.receta_id = r.id
                INNER JOIN recetas_insumos ri ON r.id = ri.receta_id
                INNER JOIN insumos i ON ri.insumo_id = i.id
                WHERE v.created_at IS NOT NULL
                    AND v.cantidad > 0
                    AND ri.cantidad > 0
                ORDER BY v.created_at DESC
            """
            
            cursor.execute(query)
            resultados = cursor.fetchall()
            
            if not resultados:
                return jsonify({
                    "error": "No se encontraron datos válidos para entrenamiento",
                    "sugerencia": "Verifica que tus recetas tengan insumos asignados con cantidad > 0"
                }), 400
            
            df = pd.DataFrame(resultados)
            print(f"✅ Registros obtenidos: {len(df)}")
            print(f"   Insumos únicos: {df['insumo'].nunique()}")
            print(f"   Recetas únicas: {df['plato'].nunique()}")
            
            # Mostrar ejemplo de datos
            print(f"\n📋 Ejemplo de datos (primeros 3):")
            for idx, row in df.head(3).iterrows():
                print(f"   {row['plato']} → {row['insumo']}: {row['cantidad_consumida']:.2f} (desperdicio: {row['desperdicio']*100:.1f}%)")

            # 3. Crear mapeos desde los datos obtenidos
            print("\n🔄 Creando mapeos...")
            
            category_mappings["INSUMO"] = {
                insumo.lower().strip(): idx 
                for idx, insumo in enumerate(df['insumo'].unique())
            }
            
            category_mappings["PLATO"] = {
                plato.lower().strip(): idx 
                for idx, plato in enumerate(df['plato'].unique())
            }
            
            category_mappings["MES"] = {
                "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
                "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
                "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
            }
            
            # PostgreSQL EXTRACT(DOW ...) devuelve: 0=Domingo, 1=Lunes, ..., 6=Sábado
            # Le sumamos 1 para que sea: 1=Domingo, 2=Lunes, ..., 7=Sábado
            category_mappings["DÍA"] = {
                "domingo": 1, "lunes": 2, "martes": 3, "miércoles": 4,
                "jueves": 5, "viernes": 6, "sábado": 7
            }
            
            print(f"✅ Mapeos creados:")
            print(f"   Insumos: {len(category_mappings['INSUMO'])}")
            print(f"   Platos/Recetas: {len(category_mappings['PLATO'])}")
            
            # Guardar mapeos
            joblib.dump(category_mappings, "category_mappings.pkl")
            print(f"✅ Mapeos guardados")

            # 4. Preparar datos para entrenamiento
            print("\n🔄 Preparando datos para el modelo...")
            X = []
            y = []
            errores = 0

            for idx, row in df.iterrows():
                try:
                    insumo_code = category_mappings["INSUMO"].get(str(row["insumo"]).lower().strip(), -1)
                    plato_code = category_mappings["PLATO"].get(str(row["plato"]).lower().strip(), -1)
                    
                    if insumo_code != -1 and plato_code != -1:
                        X.append([
                            insumo_code,           # ID del insumo
                            int(row["dia"]),       # Día de la semana (1-7)
                            int(row["mes"]),       # Mes (1-12)
                            plato_code,            # ID del plato/receta
                            float(row["desperdicio"])  # % desperdicio (0.05 = 5%)
                        ])
                        y.append(float(row["cantidad_consumida"]))  # Cantidad real consumida
                    else:
                        errores += 1
                except Exception as e:
                    errores += 1
                    if errores <= 3:
                        print(f"⚠️ Error en fila {idx}: {e}")

            if len(X) < 10:
                return jsonify({
                    "error": f"Datos insuficientes para entrenar ({len(X)} registros válidos)",
                    "minimo_requerido": 10,
                    "registros_bd": len(df),
                    "registros_con_error": errores,
                    "sugerencia": "Necesitas más ventas de recetas. Registra al menos 10 ventas diferentes."
                }), 400

            print(f"✅ Datos preparados: {len(X)} muestras válidas")
            if errores > 0:
                print(f"⚠️ {errores} registros con errores fueron omitidos")

            # 5. Entrenar modelo
            print("\n🤖 Entrenando modelo RandomForest...")
            modelo = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            modelo.fit(X, y)

            # 6. Calcular métricas
            from sklearn.metrics import r2_score, mean_absolute_error
            y_pred = modelo.predict(X)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)

            calidad = "Excelente" if r2 > 0.8 else "Bueno" if r2 > 0.6 else "Regular" if r2 > 0.4 else "Bajo"

            print(f"\n📊 Métricas del modelo:")
            print(f"   R² Score: {r2:.4f} ({calidad})")
            print(f"   Error absoluto medio: {mae:.2f} unidades")

            # 7. Guardar modelo
            joblib.dump(modelo, "modelo_consumo.pkl")
            print("✅ Modelo guardado: modelo_consumo.pkl")

            # Actualizar modelo global
            global model
            model = modelo

            # Guardar fecha de entrenamiento
            with open("ultimo_entreno.txt", "w") as f:
                f.write(datetime.now().strftime("%d/%m/%Y %H:%M"))

            print("\n✅ ¡Entrenamiento completado exitosamente!")
            print("="*60)

            return jsonify({
                "message": "Modelo entrenado exitosamente basado en ventas y recetas",
                "metricas": {
                    "registros_bd": len(df),
                    "registros_entrenamiento": len(X),
                    "registros_con_error": errores,
                    "insumos_unicos": len(category_mappings['INSUMO']),
                    "recetas_unicas": len(category_mappings['PLATO']),
                    "r2_score": round(r2, 4),
                    "precision_porcentaje": round(r2 * 100, 2),
                    "error_absoluto_medio": round(mae, 2),
                    "calidad_modelo": calidad
                },
                "recetas_disponibles": sorted(list(category_mappings["PLATO"].keys()))[:10],
                "insumos_disponibles": sorted(list(category_mappings["INSUMO"].keys()))[:10],
                "archivos_generados": ["modelo_consumo.pkl", "category_mappings.pkl"],
                "explicacion": "El modelo predice cuánto insumo se consumirá basándose en el histórico de ventas de recetas"
            }), 200

        finally:
            if conn and not conn.closed:
                conn.close()

    except Exception as e:
        print(f"❌ Error en entrenamiento: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error al entrenar el modelo: {str(e)}"}),500
# Endpoint para predecir el consumo de un plato en un mes
@app.route('/predict/month_plato', methods=['POST'])
def predict_month_plato():
    """
    Predecir consumo de un insumo para una receta específica en un mes.
    Entrada: {"insumo": "tomate", "plato": "lomo saltado", "mes": "octubre"}
    """
    try:
        data = request.get_json()

        print("\n" + "="*60)
        print("🔮 PREDICCIÓN DE CONSUMO POR RECETA")
        print("="*60)

        # Validar campos requeridos
        if not all(k in data for k in ["insumo", "plato", "mes"]):
            return jsonify({
                "error": "Faltan campos requeridos: insumo, plato, mes",
                "ejemplo": {
                    "insumo": "tomate",
                    "plato": "lomo saltado",
                    "mes": "octubre"
                }
            }), 400

        # Normalizar entradas a minúsculas
        insumo_nombre = data["insumo"].lower().strip()
        plato_nombre = data["plato"].lower().strip()
        mes_nombre = data["mes"].lower().strip()

        print(f"📥 Datos recibidos:")
        print(f"   Insumo: '{insumo_nombre}'")
        print(f"   Plato: '{plato_nombre}'")
        print(f"   Mes: '{mes_nombre}'")

        # Verificar que el modelo esté cargado
        if model is None:
            return jsonify({
                "error": "Modelo no entrenado",
                "accion": "Entrena el modelo primero usando POST /train/recetas"
            }), 400

        # Mapear valores
        insumo_code = category_mappings.get("INSUMO", {}).get(insumo_nombre, -1)
        plato_code = category_mappings.get("PLATO", {}).get(plato_nombre, -1)
        mes_code = category_mappings.get("MES", {}).get(mes_nombre, -1)

        # Validar mapeos
        if insumo_code == -1:
            insumos_disponibles = sorted(list(category_mappings.get("INSUMO", {}).keys()))
            return jsonify({
                "error": f"Insumo '{data['insumo']}' no reconocido",
                "insumos_disponibles": insumos_disponibles[:20],
                "sugerencia": "Usa uno de los insumos de la lista (en minúsculas)"
            }), 400
        
        if plato_code == -1:
            recetas_disponibles = sorted(list(category_mappings.get("PLATO", {}).keys()))
            return jsonify({
                "error": f"Receta/Plato '{data['plato']}' no reconocido",
                "recetas_disponibles": recetas_disponibles[:20],
                "sugerencia": "Usa una de las recetas de la lista (en minúsculas)"
            }), 400
        
        if mes_code == -1:
            meses_validos = list(category_mappings.get("MES", {}).keys())
            return jsonify({
                "error": f"Mes '{data['mes']}' no válido",
                "meses_validos": meses_validos,
                "sugerencia": "Usa el nombre del mes en español y minúsculas (ej: 'octubre')"
            }), 400

        print(f"✅ Mapeos encontrados:")
        print(f"   Insumo code: {insumo_code}")
        print(f"   Plato code: {plato_code}")
        print(f"   Mes code: {mes_code}")

        # Obtener unidad de medida del insumo
        conn = get_db_connection()
        unidad = "unidades"
        
        if conn:
            try:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("""
                    SELECT um.abreviatura, um.nombre
                    FROM insumos i
                    LEFT JOIN unidad_medidas um ON i.unidad_medida_id = um.id
                    WHERE LOWER(i.nombre) = %s
                """, (insumo_nombre,))
                resultado = cursor.fetchone()
                if resultado:
                    unidad = resultado['abreviatura'] or resultado['nombre'] or "unidades"
                    print(f"✅ Unidad de medida: {unidad}")
            except Exception as e:
                print(f"⚠️ No se pudo obtener unidad de medida: {e}")
            finally:
                conn.close()

        # Preparar datos para predicción
        # [insumo_code, dia, mes, plato_code, desperdicio]
        inputs = np.array([[
            insumo_code,
            4,  # Día promedio (jueves - mitad de semana)
            mes_code,
            plato_code,
            0.05  # 5% desperdicio por defecto
        ]])

        print(f"\n🤖 Realizando predicción...")
        print(f"   Inputs al modelo: {inputs[0].tolist()}")

        # Predecir
        prediction = model.predict(inputs)
        cantidad_predicha = max(0, float(prediction[0]))

        print(f"✅ Cantidad predicha: {round(cantidad_predicha, 2)} {unidad}")
        print(f"✅ Predicción completada exitosamente")
        print("="*60)

        return jsonify({
            "insumo": data["insumo"],
            "plato": data["plato"],
            "mes": data["mes"],
            "cantidad_predicha": round(cantidad_predicha, 2),
            "unidad": unidad
        }), 200

    except Exception as e:
        print(f"❌ Error en predicción: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error al realizar predicción: {str(e)}"}), 500

@app.route('/predict/month_insumo', methods=['POST'])
def predict_month_insumo():
    """
    Predecir consumo TOTAL de un insumo en un mes
    (suma de todas las recetas que lo usan)
    """
    try:
        data = request.get_json()
        
        if not all(k in data for k in ["insumo", "mes"]):
            return jsonify({"error": "Faltan campos: insumo, mes"}), 400

        insumo_nombre = data["insumo"].lower().strip()
        mes_nombre = data["mes"].lower().strip()
        
        # Verificar modelo entrenado
        if model is None:
            return jsonify({
                "error": "Modelo no entrenado",
                "accion": "Entrena con POST /train/recetas"
            }), 400
        
        # Mapear insumo y mes
        insumo_code = category_mappings.get("INSUMO", {}).get(insumo_nombre, -1)
        mes_code = category_mappings.get("MES", {}).get(mes_nombre, -1)
        
        if insumo_code == -1:
            return jsonify({
                "error": f"Insumo '{data['insumo']}' no reconocido",
                "insumos_disponibles": sorted(list(category_mappings.get("INSUMO", {}).keys()))[:20]
            }), 400
        
        if mes_code == -1:
            return jsonify({
                "error": f"Mes '{data['mes']}' no válido",
                "meses_validos": list(category_mappings.get("MES", {}).keys())
            }), 400
        
        # Obtener todas las recetas que usan este insumo
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Error de conexión a BD"}), 500
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Obtener recetas que usan el insumo
            cursor.execute("""
                SELECT DISTINCT r.nombre
                FROM recetas r
                INNER JOIN recetas_insumos ri ON r.id = ri.receta_id
                INNER JOIN insumos i ON ri.insumo_id = i.id
                WHERE LOWER(i.nombre) = %s
            """, (insumo_nombre,))
            
            recetas = cursor.fetchall()
            
            if not recetas:
                return jsonify({
                    "error": f"No hay recetas que usen '{data['insumo']}'"
                }), 404
            
            # Obtener unidad de medida
            cursor.execute("""
                SELECT um.abreviatura, um.nombre
                FROM insumos i
                LEFT JOIN unidad_medidas um ON i.unidad_medida_id = um.id
                WHERE LOWER(i.nombre) = %s
            """, (insumo_nombre,))
            
            unidad_info = cursor.fetchone()
            unidad = (unidad_info['abreviatura'] or unidad_info['nombre']) if unidad_info else "unidades"
            
            # Predecir consumo para cada receta
            consumo_total = 0
            detalles_recetas = []
            
            for receta in recetas:
                receta_nombre = receta['nombre'].lower().strip()
                plato_code = category_mappings.get("PLATO", {}).get(receta_nombre, -1)
                
                if plato_code != -1:
                    # Predecir para esta receta
                    inputs = np.array([[
                        insumo_code,
                        4,  # Día promedio
                        mes_code,
                        plato_code,
                        0.05  # 5% desperdicio
                    ]])
                    
                    pred = model.predict(inputs)[0]
                    consumo = max(0, float(pred))
                    consumo_total += consumo
                    
                    detalles_recetas.append({
                        "receta": receta['nombre'],
                        "consumo_estimado": round(consumo, 2)
                    })
            
            # Ordenar por consumo
            detalles_recetas.sort(key=lambda x: x['consumo_estimado'], reverse=True)
            
            return jsonify({
                "insumo": data["insumo"],
                "mes": data["mes"],
                "cantidad_predicha": round(consumo_total, 2),
                "unidad": unidad,
                "recetas_analizadas": len(recetas),
                "detalle_por_receta": detalles_recetas[:10]  # Top 10
            }), 200
            
        finally:
            if conn and not conn.closed:
                conn.close()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/train/sales', methods=['POST'])
def train_sales_model():
    try:
        print("🔄 Iniciando entrenamiento del modelo de ventas...")
        
        # Obtener datos de ventas
        df = get_sales_training_data()
        if df is None or df.empty:
            print("❌ No hay datos de ventas disponibles")
            return jsonify({"error": "No hay datos de ventas disponibles para entrenamiento"}), 400

        print(f"✅ Datos obtenidos: {len(df)} registros")
        print(f"📊 Productos únicos: {df['producto'].nunique()}")

        # IMPORTANTE: Normalizar nombres de productos a minúsculas
        print("\n🔄 Normalizando nombres de productos...")
        df['producto'] = df['producto'].str.lower().str.strip()
        
        print(f"✅ Productos después de normalizar: {df['producto'].unique().tolist()}")

        # Preparar datos - SOLO PARA PREDICCIÓN DE VENTAS
        le_productos = LabelEncoder()
        df['producto_encoded'] = le_productos.fit_transform(df['producto'])
        
        # Guardar lista de productos conocidos (ya en minúsculas)
        productos_conocidos = df['producto'].unique().tolist()
        joblib.dump(productos_conocidos, "productos_conocidos.pkl")
        print(f"✅ Productos conocidos guardados: {productos_conocidos}")
        
        # X e y SOLO para predicción de cantidad de ventas
        X = df[['producto_encoded', 'mes', 'año']].values
        y = df['ventas'].values

        # Entrenar modelo
        print("\n🤖 Entrenando modelo RandomForest...")
        new_model = RandomForestRegressor(n_estimators=100, random_state=42)
        new_model.fit(X, y)
        print("✅ Modelo entrenado correctamente")

        # Guardar modelo y encoder
        joblib.dump(new_model, "modelo_ventas.pkl")
        joblib.dump(le_productos, "label_encoder_productos.pkl")
        print("✅ Archivos guardados:")
        print("   - modelo_ventas.pkl")
        print("   - label_encoder_productos.pkl")
        print("   - productos_conocidos.pkl")
        
        print(f"\n✅ Clases del encoder: {list(le_productos.classes_)}")

        # Actualizar variable global
        global sales_model
        sales_model = new_model

        # Información adicional sobre precios
        precios_info = {
            "precio_minimo": float(df['precio_venta'].min()),
            "precio_maximo": float(df['precio_venta'].max()),
            "precio_promedio": float(df['precio_venta'].mean())
        }

        return jsonify({
            "message": "Modelo de ventas entrenado exitosamente",
            "n_samples": len(df),
            "productos_unicos": len(df['producto'].unique()),
            "productos": productos_conocidos,
            "precios_info": precios_info,
            "archivos_generados": [
                "modelo_ventas.pkl",
                "label_encoder_productos.pkl",
                "productos_conocidos.pkl"
            ]
        }), 200

    except Exception as e:
        print(f"❌ Error en entrenamiento: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error al entrenar el modelo: {str(e)}"}), 500

@app.route('/predict/sales', methods=['POST'])
@app.route('/predict/sales', methods=['POST'])
def predict_sales():
    """
    Predecir ventas e ingresos en Bolivianos
    Soporta predicción por: día, semana o mes
    """
    try:
        data = request.get_json()
        
        print("\n" + "="*60)
        print("🔮 PREDICCIÓN DE VENTAS E INGRESOS")
        print("="*60)
        
        # Validar campos requeridos
        required_fields = ["producto", "mes_inicial", "cantidad_periodos"]
        if not all(k in data for k in required_fields):
            return jsonify({
                "error": "Faltan campos requeridos: producto, mes_inicial, cantidad_periodos",
                "ejemplo": {
                    "producto": "Paella Valenciana",
                    "mes_inicial": "octubre",
                    "cantidad_periodos": 3,
                    "tipo_periodo": "mes"
                }
            }), 400

        # Tipo de periodo (día, semana, mes)
        tipo_periodo = data.get("tipo_periodo", "mes").lower()
        if tipo_periodo not in ["dia", "semana", "mes"]:
            return jsonify({
                "error": "tipo_periodo debe ser: 'dia', 'semana' o 'mes'",
                "recibido": tipo_periodo
            }), 400

        print(f"📥 Datos recibidos:")
        print(f"   Producto: '{data['producto']}'")
        print(f"   Mes inicial: '{data['mes_inicial']}'")
        print(f"   Cantidad de periodos: {data['cantidad_periodos']}")
        print(f"   Tipo de periodo: {tipo_periodo}")

        # Cargar el modelo y el encoder
        try:
            modelo_ventas = joblib.load("modelo_ventas.pkl")
            le_productos = joblib.load("label_encoder_productos.pkl")
            productos_conocidos = joblib.load("productos_conocidos.pkl")
            print(f"✅ Modelos cargados correctamente")
        except FileNotFoundError as e:
            return jsonify({
                "error": "Modelo de ventas no entrenado",
                "accion": "Entrena el modelo con POST /train/sales"
            }), 400

        # Normalizar producto
        producto_input = data["producto"].lower().strip()
        
        if producto_input not in productos_conocidos:
            return jsonify({
                "error": f"Producto '{data['producto']}' no reconocido",
                "productos_disponibles": productos_conocidos
            }), 400

        # Validar mes
        mes_inicial_str = data["mes_inicial"].lower().strip()
        MESES_MAP = {
            "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
            "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
            "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
        }
        MESES_ORDEN = list(MESES_MAP.keys())
        
        if mes_inicial_str not in MESES_MAP:
            return jsonify({"error": f"Mes '{data['mes_inicial']}' no válido"}), 400
        
        mes_inicial = MESES_MAP[mes_inicial_str]

        # Validar cantidad de periodos
        try:
            cantidad_periodos = int(data["cantidad_periodos"])
            max_periodos = {"dia": 90, "semana": 52, "mes": 24}
            if cantidad_periodos < 1 or cantidad_periodos > max_periodos[tipo_periodo]:
                return jsonify({
                    "error": f"Para '{tipo_periodo}', cantidad debe estar entre 1 y {max_periodos[tipo_periodo]}"
                }), 400
        except (ValueError, TypeError):
            return jsonify({"error": "cantidad_periodos debe ser un número entero"}), 400

        # Codificar producto
        try:
            producto_encoded = le_productos.transform([producto_input])[0]
        except ValueError:
            return jsonify({"error": "Error al codificar producto"}), 500

        # OBTENER PRECIO DE VENTA
        print(f"\n💰 Obteniendo precio de venta...")
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Error de conexión a BD"}), 500
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT 
                    r.nombre,
                    AVG(v.precio) as precio_venta
                FROM ventas v
                INNER JOIN recetas r ON v.receta_id = r.id
                WHERE LOWER(r.nombre) = %s
                    AND v.precio > 0
                    AND v.cantidad > 0
                GROUP BY r.id, r.nombre
            """
            
            cursor.execute(query, (producto_input,))
            resultado = cursor.fetchone()
            
            if not resultado:
                return jsonify({
                    "error": f"No se encontraron datos para '{data['producto']}'"
                }), 400
            
            precio_venta = float(resultado['precio_venta']) if resultado['precio_venta'] else 0
            print(f"✅ Precio promedio: {precio_venta:.2f} Bs")
            
            if precio_venta == 0:
                return jsonify({"error": "Precio de venta no válido"}), 400
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if conn and not conn.closed:
                conn.close()

        # REALIZAR PREDICCIONES SEGÚN TIPO DE PERIODO
        print(f"\n🤖 Generando {cantidad_periodos} predicciones por {tipo_periodo}...")
        
        predicciones_ventas = []
        predicciones_ingresos = []
        detalles = []
        año_actual = datetime.now().year
        
        # Factor de conversión según periodo
        factores = {
            "dia": 1/30,      # Un día es 1/30 de un mes
            "semana": 1/4,    # Una semana es 1/4 de un mes
            "mes": 1          # Un mes es 1 mes
        }
        factor = factores[tipo_periodo]
        
        total_ventas = 0
        total_ingresos = 0

        for i in range(cantidad_periodos):
            # Calcular periodo actual
            if tipo_periodo == "mes":
                mes = ((mes_inicial - 1 + i) % 12) + 1
                año = año_actual + ((mes_inicial - 1 + i) // 12)
                periodo_label = f"{MESES_ORDEN[mes - 1].capitalize()} {año}"
            elif tipo_periodo == "semana":
                semanas_totales = i
                mes = ((mes_inicial - 1 + (semanas_totales // 4)) % 12) + 1
                año = año_actual + ((mes_inicial - 1 + (semanas_totales // 4)) // 12)
                semana_del_mes = (semanas_totales % 4) + 1
                periodo_label = f"Semana {semana_del_mes} - {MESES_ORDEN[mes - 1].capitalize()} {año}"
            else:  # dia
                dias_totales = i
                mes = ((mes_inicial - 1 + (dias_totales // 30)) % 12) + 1
                año = año_actual + ((mes_inicial - 1 + (dias_totales // 30)) // 12)
                dia = (dias_totales % 30) + 1
                periodo_label = f"Día {dia} - {MESES_ORDEN[mes - 1].capitalize()} {año}"
            
            # Predecir ventas del mes y ajustar según periodo
            X_pred = np.array([[producto_encoded, mes, año]])
            pred_ventas_mes = float(modelo_ventas.predict(X_pred)[0])
            pred_ventas = max(0, round(pred_ventas_mes * factor, 2))
            
            # Calcular ingresos
            ingresos = pred_ventas * precio_venta
            
            total_ventas += pred_ventas
            total_ingresos += ingresos
            
            predicciones_ventas.append(pred_ventas)
            predicciones_ingresos.append(round(ingresos, 2))
            
            detalles.append({
                "periodo": periodo_label,
                "numero_periodo": i + 1,
                "ventas_predichas": round(pred_ventas, 2),
                "ingresos_estimados": round(ingresos, 2)
            })
            
            print(f"   📅 {periodo_label}: {pred_ventas:.1f} platos = {ingresos:.2f} Bs")

        # Estadísticas
        promedio_ventas = total_ventas / cantidad_periodos if cantidad_periodos > 0 else 0
        promedio_ingresos = total_ingresos / cantidad_periodos if cantidad_periodos > 0 else 0
        
        print(f"\n📊 RESUMEN:")
        print(f"   Total Ventas: {total_ventas:.0f} platos")
        print(f"   Total Ingresos: {total_ingresos:.2f} Bs")

        return jsonify({
            "producto": data["producto"],
            "producto_normalizado": producto_input,
            "mes_inicial": mes_inicial_str,
            "tipo_periodo": tipo_periodo,
            "cantidad_periodos": cantidad_periodos,
            "precio_unitario": round(precio_venta, 2),
            "predicciones": predicciones_ventas,
            "predicciones_ingresos": predicciones_ingresos,
            "detalles": detalles,
            "estadisticas": {
                "total_ventas": round(total_ventas, 2),
                "total_ingresos": round(total_ingresos, 2),
                "promedio_ventas": round(promedio_ventas, 2),
                "promedio_ingresos": round(promedio_ingresos, 2),
                "minimo_ventas": round(min(predicciones_ventas), 2) if predicciones_ventas else 0,
                "maximo_ventas": round(max(predicciones_ventas), 2) if predicciones_ventas else 0,
                "minimo_ingresos": round(min(predicciones_ingresos), 2) if predicciones_ingresos else 0,
                "maximo_ingresos": round(max(predicciones_ingresos), 2) if predicciones_ingresos else 0,
                "moneda": "Bs."
            }
        }), 200

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/get/insumos_disponibles', methods=['GET'])
def get_insumos_disponibles():
    """
    Obtener lista de todos los insumos disponibles en la base de datos
    con información adicional sobre recetas que los usan
    """
    try:
        print("\n" + "="*60)
        print("📋 OBTENIENDO INSUMOS DISPONIBLES")
        print("="*60)
        
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Error de conexión a la base de datos"}), 500

        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Consulta principal: todos los insumos con información adicional
            query = """
                SELECT 
                    i.id,
                    i.nombre as insumo,
                    COALESCE(um.abreviatura, um.nombre, 'unidades') as unidad,
                    COUNT(DISTINCT ri.receta_id) as recetas_que_lo_usan,
                    ARRAY_AGG(DISTINCT r.nombre) FILTER (WHERE r.nombre IS NOT NULL) as recetas,
                    COALESCE(AVG(ri.cantidad), 0) as cantidad_promedio_por_receta,
                    COALESCE(AVG(ri.desperdicio), 0) as desperdicio_promedio
                FROM insumos i
                LEFT JOIN unidad_medidas um ON i.unidad_medida_id = um.id
                LEFT JOIN recetas_insumos ri ON i.id = ri.insumo_id
                LEFT JOIN recetas r ON ri.receta_id = r.id
                GROUP BY i.id, i.nombre, um.abreviatura, um.nombre
                ORDER BY recetas_que_lo_usan DESC, i.nombre ASC
            """
            
            cursor.execute(query)
            resultados = cursor.fetchall()
            
            if not resultados:
                return jsonify({
                    "message": "No hay insumos registrados en la base de datos",
                    "insumos_disponibles": [],
                    "total_insumos": 0
                }), 200
            
            # Procesar resultados
            insumos_con_recetas = []
            insumos_sin_recetas = []
            
            for row in resultados:
                insumo_info = {
                    "id": row['id'],
                    "nombre": row['insumo'],
                    "nombre_normalizado": row['insumo'].lower().strip(),
                    "unidad": row['unidad'],
                    "recetas_que_lo_usan": row['recetas_que_lo_usan'],
                    "cantidad_promedio": round(float(row['cantidad_promedio_por_receta']), 2) if row['cantidad_promedio_por_receta'] else 0,
                    "desperdicio_promedio": round(float(row['desperdicio_promedio']), 2) if row['desperdicio_promedio'] else 0
                }
                
                if row['recetas_que_lo_usan'] > 0:
                    insumo_info['recetas'] = row['recetas'][:5]  # Máximo 5 recetas por insumo
                    insumos_con_recetas.append(insumo_info)
                else:
                    insumo_info['recetas'] = []
                    insumos_sin_recetas.append(insumo_info)
            
            # Verificar modelo entrenado para ver qué insumos están mapeados
            insumos_en_modelo = []
            if 'INSUMO' in category_mappings:
                insumos_en_modelo = sorted(list(category_mappings['INSUMO'].keys()))
            
            print(f"✅ Insumos encontrados: {len(resultados)}")
            print(f"   Con recetas: {len(insumos_con_recetas)}")
            print(f"   Sin recetas: {len(insumos_sin_recetas)}")
            print(f"   En modelo entrenado: {len(insumos_en_modelo)}")
            
            return jsonify({
                "total_insumos": len(resultados),
                "insumos_con_recetas": insumos_con_recetas,
                "insumos_sin_recetas": insumos_sin_recetas,
                "insumos_en_modelo_entrenado": insumos_en_modelo,
                "resumen": {
                    "total": len(resultados),
                    "con_recetas": len(insumos_con_recetas),
                    "sin_recetas": len(insumos_sin_recetas),
                    "disponibles_para_prediccion": len(insumos_en_modelo)
                },
                "nota": {
                    "uso": "Para usar en /predict/month_insumo, usa el 'nombre_normalizado' (en minúsculas)",
                    "ejemplo": "POST /predict/month_insumo con {\"insumo\": \"tomate\", \"mes\": \"octubre\"}"
                }
            }), 200

        finally:
            if conn and not conn.closed:
                conn.close()

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error al obtener insumos: {str(e)}"}), 500

@app.route('/get/recetas_disponibles', methods=['GET'])
def get_recetas_disponibles():
    """
    Obtener lista de todas las recetas disponibles con sus insumos
    """
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Error de conexión a la base de datos"}), 500

        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT 
                    r.id,
                    r.nombre as receta,
                    COUNT(ri.insumo_id) as total_insumos,
                    ARRAY_AGG(
                        json_build_object(
                            'insumo', i.nombre,
                            'cantidad', ri.cantidad,
                            'desperdicio', ri.desperdicio,
                            'unidad', COALESCE(um.abreviatura, um.nombre, 'unidades')
                        )
                    ) as insumos
                FROM recetas r
                LEFT JOIN recetas_insumos ri ON r.id = ri.receta_id
                LEFT JOIN insumos i ON ri.insumo_id = i.id
                LEFT JOIN unidad_medidas um ON i.unidad_medida_id = um.id
                GROUP BY r.id, r.nombre
                ORDER BY total_insumos DESC, r.nombre ASC
            """
            
            cursor.execute(query)
            resultados = cursor.fetchall()
            
            recetas_con_insumos = []
            recetas_sin_insumos = []
            
            for row in resultados:
                receta_info = {
                    "id": row['id'],
                    "nombre": row['receta'],
                    "nombre_normalizado": row['receta'].lower().strip(),
                    "total_insumos": row['total_insumos'],
                    "insumos": row['insumos'] if row['insumos'][0] else []
                }
                
                if row['total_insumos'] > 0:
                    recetas_con_insumos.append(receta_info)
                else:
                    recetas_sin_insumos.append(receta_info)
            
            # Verificar recetas en modelo entrenado
            recetas_en_modelo = []
            if 'PLATO' in category_mappings:
                recetas_en_modelo = sorted(list(category_mappings['PLATO'].keys()))
            
            return jsonify({
                "total_recetas": len(resultados),
                "recetas_con_insumos": recetas_con_insumos,
                "recetas_sin_insumos": recetas_sin_insumos,
                "recetas_en_modelo_entrenado": recetas_en_modelo,
                "resumen": {
                    "total": len(resultados),
                    "con_insumos": len(recetas_con_insumos),
                    "sin_insumos": len(recetas_sin_insumos),
                    "disponibles_para_prediccion": len(recetas_en_modelo)
                }
            }), 200

        finally:
            if conn and not conn.closed:
                conn.close()

    except Exception as e:
        return jsonify({"error": f"Error al obtener recetas: {str(e)}"}), 500
        
if __name__ == '__main__':
    # Inicializar category_mappings si no existe
    if 'category_mappings' not in globals() or not category_mappings:
        category_mappings = {}
    
    # Actualizar mapeos al iniciar
    print("🔄 Actualizando mapeos de la base de datos...")
    if update_category_mappings():
        print("✅ Mapeos actualizados correctamente")
    else:
        print("⚠️ No se pudieron actualizar los mapeos")
    
    # Intentar cargar los modelos
    try:
        model = joblib.load("modelo_consumo.pkl")
        print("✅ Modelo de consumo cargado")
    except FileNotFoundError:
        print("⚠️ Modelo de consumo no encontrado. Intentando entrenar automáticamente...")
        # Entrenar modelo automáticamente al iniciar
        try:
            print("🤖 Iniciando entrenamiento automático...")
            # Simular una request para entrenar
            from unittest.mock import Mock
            mock_request = Mock()
            mock_request.get_json.return_value = {}
            
            # Entrenar usando la función directamente
            conn = get_db_connection()
            if conn:
                conn.close()
                # Llamar a la lógica de entrenamiento
                print("🔄 Entrenamiento automático iniciado...")
                # Aquí llamarías a train_recetas() pero sin Flask request
                print("✅ Para entrenar manualmente: POST /train/recetas")
            model = None
        except Exception as e:
            print(f"❌ Error en entrenamiento automático: {e}")
            model = None
    except Exception as e:
        print(f"❌ Error cargando modelo de consumo: {e}")
        model = None
    
    try:
        sales_model = joblib.load("modelo_ventas.pkl")
        print("✅ Modelo de ventas cargado")
    except FileNotFoundError:
        print("⚠️ Modelo de ventas no encontrado. Entrénalo con POST /train/sales")
        sales_model = None
    except Exception as e:
        print(f"❌ Error cargando modelo de ventas: {e}")
        sales_model = None
    
    print("\n🚀 Servidor Flask iniciando...")
    print("📋 Endpoints disponibles:")
    print("   GET  /get/insumos_disponibles")
    print("   GET  /get/recetas_disponibles") 
    print("   POST /train/recetas ⭐ (ENTRENA EL MODELO)")
    print("   POST /predict/month_insumo")
    print("   POST /predict/month_plato")
    print("\n⚠️ IMPORTANTE: Si /predict/month_insumo falla:")
    print("   1. Ejecuta POST /train/recetas primero")
    print("   2. Luego usa POST /predict/month_insumo")
    print("="*50)
    
    # Configurar Flask con variables de entorno
    app.config.update(FLASK_CONFIG)
    app.run(debug=FLASK_CONFIG['DEBUG'], port=FLASK_CONFIG['PORT'])