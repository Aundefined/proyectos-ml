from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import pickle
import tensorflow as tf
import os
from sklearn.metrics.pairwise import cosine_similarity  

from . import recomendador_peliculas_bp

# Variables globales para el modelo y datos
modelo = None
modelo_embeddings = None
ratings = None
movies = None
user_to_index = None
movie_to_index = None
user_embeddings = None
movie_embeddings = None


def cargar_modelo_y_datos():
    """Carga el modelo entrenado y los datos necesarios"""
    global modelo, modelo_embeddings, ratings, movies, user_to_index, movie_to_index, user_embeddings, movie_embeddings
    
    try:
        print("🔄 Intentando cargar modelo...")
        
        # Cargar modelo
        modelo = tf.keras.models.load_model('ml-models/modelo_recomendacion.h5', compile=False)
        print("✓ Modelo cargado sin compilar")
        
        # Recompilar manualmente
        modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print("✓ Modelo recompilado")
        
          # Cargar modelo embeddings
        modelo_embeddings = tf.keras.models.load_model('ml-models/modelo_embeddings.h5', compile=False)
        print("✓ Modelo embeddings cargado sin compilar")
        
        # Recompilar manualmente
        modelo_embeddings.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print("✓ Modelo embeddings recompilado")
        
        # Cargar datos
        ratings = pd.read_csv('attached-files/ratings.csv')
        movies = pd.read_csv('attached-files/movies.csv')
        print("✓ CSVs cargados")
        
        # Cargar mapeos
        with open('attached-files/mapeos_recomendacion.pkl', 'rb') as f:
            mapeos = pickle.load(f)
            user_to_index = mapeos['user_to_index']
            movie_to_index = mapeos['movie_to_index']
        print("✓ Mapeos cargados")
        
        # Extraer embeddings del modelo - AQUÍ ESTÁ EL PROBLEMA
        print("🔍 Capas del modelo:")
        for i, layer in enumerate(modelo.layers):
            print(f"  {i}: {layer.name} - {type(layer).__name__}")
        
        # Intentar diferentes nombres de capas
        user_embedding_layer = None
        movie_embedding_layer = None
        
        for layer in modelo.layers:
            if 'embedding' in layer.name.lower():
                if user_embedding_layer is None:
                    user_embedding_layer = layer
                    print(f"✓ User embedding layer: {layer.name}")
                elif movie_embedding_layer is None:
                    movie_embedding_layer = layer
                    print(f"✓ Movie embedding layer: {layer.name}")
        
        if user_embedding_layer is not None:
            user_embeddings = user_embedding_layer.get_weights()[0]
            print(f"✓ User embeddings extraídos: {user_embeddings.shape}")
        else:
            print("❌ No se encontró capa de user embedding")
            
        if movie_embedding_layer is not None:
            movie_embeddings = movie_embedding_layer.get_weights()[0]
            print(f"✓ Movie embeddings extraídos: {movie_embeddings.shape}")
        else:
            print("❌ No se encontró capa de movie embedding")
        
        print("🎉 TODO CARGADO CORRECTAMENTE")
        return True
    except Exception as e:
        print(f"💥 ERROR AL CARGAR: {e}")
        import traceback
        traceback.print_exc()
        return False

def obtener_generos_disponibles():
    """Obtiene todos los géneros individuales únicos disponibles en el dataset"""
    try:
        if movies is None:
            raise Exception("Datos de películas no disponibles")
        
        # Set para almacenar géneros únicos
        generos_individuales = set()
        
        # Procesar cada fila de géneros
        for genres_string in movies['genres'].dropna():
            if pd.notna(genres_string) and genres_string.strip():
                # Dividir por | y agregar cada género individual
                generos_en_fila = genres_string.split('|')
                for genero in generos_en_fila:
                    genero_limpio = genero.strip()
                    if genero_limpio and genero_limpio != '(no genres listed)':
                        generos_individuales.add(genero_limpio)
        
        # Convertir a lista y ordenar alfabéticamente
        generos_unicos = sorted(list(generos_individuales))
        
        print(f"📊 Géneros individuales únicos encontrados: {len(generos_unicos)}")
        print(f"🎬 Géneros: {generos_unicos[:10]}...") # Mostrar los primeros 10
        return generos_unicos
        
    except Exception as e:
        print(f"💥 Error en obtener_generos_disponibles: {e}")
        return []

def obtener_peliculas_onboarding(generos_seleccionados):
    """Obtiene películas aleatorias basadas en los géneros seleccionados"""
    try:
        # Verificar que tenemos datos
        if ratings is None or movies is None:
            raise Exception("Datos no disponibles")
        
        if not generos_seleccionados:
            raise Exception("No se seleccionaron géneros")
        
        print(f"📊 Géneros seleccionados: {generos_seleccionados}")
        
        peliculas_seleccionadas = []
        
        for genero in generos_seleccionados:
            print(f"🎬 Procesando género: {genero}")
            
            # Buscar películas que contengan este género en su string de géneros
            # Usamos str.contains con regex para buscar el género exacto
            patron = f'(^|\\|){genero}(\\||$)'
            peliculas_del_genero = movies[movies['genres'].str.contains(patron, na=False, regex=True)].copy()
            
            print(f"📊 Películas encontradas para {genero}: {len(peliculas_del_genero)}")
            
            if len(peliculas_del_genero) == 0:
                print(f"⚠️ No se encontraron películas para el género {genero}")
                continue
            
            # Tomar máximo 5 películas aleatorias (o todas si hay menos de 5)
            n_peliculas = min(10, len(peliculas_del_genero))
            peliculas_aleatorias = peliculas_del_genero.sample(n=n_peliculas, random_state=None)
            
            # Agregar a la lista de seleccionadas
            for _, movie in peliculas_aleatorias.iterrows():
                peliculas_seleccionadas.append({
                    'movieId': int(movie['movieId']),
                    'title': movie['title'],
                    'genres': movie['genres']
                })
            
            print(f"✅ Agregadas {n_peliculas} películas de {genero}")
        
        print(f"📊 Total películas seleccionadas: {len(peliculas_seleccionadas)}")
        return peliculas_seleccionadas
        
    except Exception as e:
        print(f"💥 Error en obtener_peliculas_onboarding: {e}")
        import traceback
        traceback.print_exc()
        return []

def crear_embedding_usuario_nuevo(ratings_onboarding):
    """Crea un embedding real para un usuario nuevo usando optimización con el modelo entrenado"""
    try:
        import tensorflow as tf
        from tensorflow.keras.optimizers import Adam
        
        print(f"🧠 Creando embedding real para usuario nuevo con {len(ratings_onboarding)} ratings...")
        
        if modelo is None or movie_embeddings is None:
            raise Exception("Modelo o embeddings no disponibles")
        
        # Preparar datos de entrenamiento
        movie_ids = []
        user_ratings = []
        movie_embedding_vectors = []
        
        for movie_id, rating in ratings_onboarding.items():
            if movie_id in movie_to_index:
                movie_idx = movie_to_index[movie_id]
                if movie_idx < len(movie_embeddings):
                    movie_ids.append(movie_id)
                    user_ratings.append(rating)
                    movie_embedding_vectors.append(movie_embeddings[movie_idx])
        
        if len(movie_ids) < 3:
            print("⚠️ Muy pocas películas válidas, usando embedding promedio")
            return np.mean(user_embeddings, axis=0)
        
        print(f"📊 Optimizando con {len(movie_ids)} películas válidas")
        
        # Convertir a arrays de TensorFlow
        movie_embeddings_tensor = tf.constant(movie_embedding_vectors, dtype=tf.float32)
        target_ratings = tf.constant(user_ratings, dtype=tf.float32)
        
        # Crear variable de embedding de usuario a optimizar
        # Inicializar con promedio de embeddings existentes + ruido pequeño
        initial_embedding = np.mean(user_embeddings, axis=0) + np.random.normal(0, 0.01, 50)
        user_embedding_var = tf.Variable(initial_embedding, dtype=tf.float32, trainable=True)
        
        # Crear modelo de predicción directo
        def predict_ratings(user_emb, movie_embs):
            """Predice ratings usando la misma arquitectura que el modelo original"""
            # Expandir user embedding para hacer broadcast
            user_emb_expanded = tf.expand_dims(user_emb, 0)  # (1, 50)
            user_emb_repeated = tf.repeat(user_emb_expanded, tf.shape(movie_embs)[0], axis=0)  # (n_movies, 50)
            
            # Concatenar user y movie embeddings
            concat_features = tf.concat([user_emb_repeated, movie_embs], axis=1)  # (n_movies, 100)
            
            # Aplicar capas densas del modelo original (copiamos la arquitectura)
            # Dense layer 1: 100 -> 16
            dense1_weights = modelo.layers[7].get_weights()[0]  # (100, 16)
            dense1_bias = modelo.layers[7].get_weights()[1]     # (16,)
            x = tf.nn.relu(tf.matmul(concat_features, dense1_weights) + dense1_bias)
            
            # Dropout simulation (sin dropout en inferencia)
            
            # Dense layer 2: 16 -> 16  
            dense2_weights = modelo.layers[9].get_weights()[0]  # (16, 16)
            dense2_bias = modelo.layers[9].get_weights()[1]     # (16,)
            x = tf.nn.relu(tf.matmul(x, dense2_weights) + dense2_bias)
            
            # Output layer: 16 -> 1
            output_weights = modelo.layers[11].get_weights()[0]  # (16, 1)
            output_bias = modelo.layers[11].get_weights()[1]     # (1,)
            predictions = tf.matmul(x, output_weights) + output_bias
            
            return tf.squeeze(predictions)  # (n_movies,)
        
        # Función de pérdida
        def loss_function():
            predictions = predict_ratings(user_embedding_var, movie_embeddings_tensor)
            mse_loss = tf.reduce_mean(tf.square(predictions - target_ratings))
            
            # Regularización L2 para evitar embeddings extremos
            l2_reg = 0.01 * tf.reduce_sum(tf.square(user_embedding_var))
            
            return mse_loss + l2_reg
        
        # Optimizador
        optimizer = Adam(learning_rate=0.01)
        
        # Entrenamiento
        print("🔧 Optimizando embedding...")
        for epoch in range(200):
            with tf.GradientTape() as tape:
                loss = loss_function()
            
            # Calcular gradientes y aplicar
            gradients = tape.gradient(loss, [user_embedding_var])
            optimizer.apply_gradients(zip(gradients, [user_embedding_var]))
            
            # Log progreso cada 50 epochs
            if epoch % 50 == 0:
                predictions = predict_ratings(user_embedding_var, movie_embeddings_tensor)
                mae = tf.reduce_mean(tf.abs(predictions - target_ratings))
                print(f"  Epoch {epoch}: Loss={loss:.4f}, MAE={mae:.4f}")
        
        # Resultado final
        final_embedding = user_embedding_var.numpy()
        final_predictions = predict_ratings(user_embedding_var, movie_embeddings_tensor).numpy()
        final_mae = np.mean(np.abs(final_predictions - user_ratings))
        
        print(f"✅ Embedding optimizado completado!")
        print(f"📊 MAE final: {final_mae:.4f}")
        print(f"📊 Ratings reales: {user_ratings}")
        print(f"📊 Predicciones: {final_predictions}")
        
        return final_embedding
        
    except Exception as e:
        print(f"💥 Error en crear_embedding_usuario_nuevo: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: usar embedding promedio
        return np.mean(user_embeddings, axis=0) if user_embeddings is not None else np.zeros(50)

def generar_recomendaciones(embedding_usuario, generos_seleccionados, n_recomendaciones=10):
    """Genera recomendaciones usando el modelo entrenado con predicciones reales"""
    try:
        print(f"🎯 Generando recomendaciones para géneros: {generos_seleccionados}...")
        
        predicciones = []
        peliculas_candidatas = []
        
        # Obtener 100 películas aleatorias de cada género seleccionado
        for genero in generos_seleccionados:
            print(f"🎬 Buscando películas de género: {genero}")
            
            # Buscar películas que contengan este género
            patron = f'(^|\\|){genero}(\\||$)'
            peliculas_del_genero = movies[movies['genres'].str.contains(patron, na=False, regex=True)].copy()
            
            print(f"📊 Películas encontradas para {genero}: {len(peliculas_del_genero)}")
            
            if len(peliculas_del_genero) > 0:
                # Tomar máximo 100 películas aleatorias
                n_peliculas = min(100, len(peliculas_del_genero))
                peliculas_aleatorias = peliculas_del_genero.sample(n=n_peliculas, random_state=None)
                peliculas_candidatas.append(peliculas_aleatorias)
        
        # Combinar todas las películas candidatas y eliminar duplicados
        if peliculas_candidatas:
            todas_candidatas = pd.concat(peliculas_candidatas, ignore_index=True)
            todas_candidatas = todas_candidatas.drop_duplicates(subset=['movieId'], keep='first')
            print(f"📊 Total películas candidatas (sin duplicados): {len(todas_candidatas)}")
        else:
            print("⚠️ No se encontraron películas candidatas")
            return []
        
        # Debug: verificar shapes
        print(f"🔍 embedding_usuario shape: {embedding_usuario.shape}")
        print(f"🔍 modelo_embeddings disponible: {modelo_embeddings is not None}")
        
        # Generar predicciones para las películas candidatas
        for i, (_, movie_row) in enumerate(todas_candidatas.iterrows()):
            try:
                movie_id = movie_row['movieId']
                
                if movie_id not in movie_to_index:
                    continue
                    
                movie_idx = movie_to_index[movie_id]
                if movie_idx >= len(movie_embeddings):
                    continue
                
                movie_embedding = movie_embeddings[movie_idx]
            
                
                # Usar el modelo embeddings cargado para predecir
                prediction = modelo_embeddings.predict([
                    embedding_usuario.reshape(1,-1), 
                    movie_embedding.reshape(1,-1)   
                ], verbose=0)[0][0]
                
                if i < 3:  # Debug para las primeras 3
                    print(f"🔍 Prediction para movie {i}: {prediction}")
                
                predicted_rating = max(1.0, min(5.0, prediction))
                
                predicciones.append({
                    'movieId': int(movie_id),
                    'predicted_rating': float(predicted_rating),
                    'title': movie_row['title'],
                    'genres': movie_row['genres']
                })
                
            except Exception as e:
                print(f"💥 Error en predicción movie {movie_id}: {e}")
                continue
        
        print(f"✅ Total predicciones calculadas: {len(predicciones)}")
        
        # Debug: mostrar variedad de predicciones
        if predicciones:
            ratings_unicos = set(p['predicted_rating'] for p in predicciones)
            print(f"🔍 Ratings únicos generados: {len(ratings_unicos)}")
            print(f"🔍 Rango de ratings: {min(p['predicted_rating'] for p in predicciones):.3f} - {max(p['predicted_rating'] for p in predicciones):.3f}")
        
        # Ordenar por rating predicho
        predicciones.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        return predicciones[:n_recomendaciones]
        
    except Exception as e:
        print(f"💥 Error en generar_recomendaciones: {e}")
        return []
    


@recomendador_peliculas_bp.route('/')
def index():
    """Página principal del recomendador"""
    return render_template('recomendador_peliculas.html')


@recomendador_peliculas_bp.route('/seleccionar-generos')
def seleccionar_generos():
    """Página para seleccionar géneros"""
    global modelo, ratings, movies
    
    # Cargar modelo si no está cargado
    if modelo is None:
        print("Modelo es None, cargando...")
        cargar_modelo_y_datos()
    
    if not modelo:
        return "<h1>Error: No se pudo cargar el modelo</h1>"
    
    # Obtener géneros disponibles
    generos = obtener_generos_disponibles()
    return render_template('seleccionar_generos.html', generos=generos)


@recomendador_peliculas_bp.route('/onboarding', methods=['POST'])
def onboarding():
    """Página de onboarding con películas basadas en géneros seleccionados"""
    try:
        # Obtener géneros seleccionados del formulario
        generos_seleccionados = request.form.getlist('generos')
        
        if not generos_seleccionados or len(generos_seleccionados) < 1:
            # Redirigir de vuelta a selección de géneros con error
            generos = obtener_generos_disponibles()
            return render_template('seleccionar_generos.html', 
                                 generos=generos,
                                 error="Por favor, selecciona al menos un género")
        
        print(f"📝 Géneros seleccionados: {generos_seleccionados}")
        
        # Obtener películas basadas en géneros
        peliculas = obtener_peliculas_onboarding(generos_seleccionados)
        
        if not peliculas:
            generos = obtener_generos_disponibles()
            return render_template('seleccionar_generos.html', 
                                 generos=generos,
                                 error="No se encontraron películas para los géneros seleccionados")
        
        return render_template('onboarding.html', 
                             peliculas=peliculas,
                             generos_seleccionados=generos_seleccionados)
        
    except Exception as e:
        print(f"💥 Error en onboarding: {e}")
        generos = obtener_generos_disponibles()
        return render_template('seleccionar_generos.html', 
                             generos=generos,
                             error=f"Error al cargar películas: {str(e)}")


@recomendador_peliculas_bp.route('/procesar_onboarding', methods=['POST'])
def procesar_onboarding():
    """Procesa las calificaciones del onboarding y genera recomendaciones"""
    try:
        print("🚀 Iniciando procesamiento...")
        
        # Obtener ratings del formulario
        ratings_onboarding = {}
        generos_str = request.form.get('generos_seleccionados', '')
        generos_seleccionados = [g.strip() for g in generos_str.split(',') if g.strip()]
        
        for key, value in request.form.items():
            if key.startswith('rating_') and value:
                movie_id = int(key.replace('rating_', ''))
                ratings_onboarding[movie_id] = float(value)
        
        print(f"📝 Ratings recibidos: {ratings_onboarding}")
    
        # Crear embedding para usuario nuevo
        print("🧠 Creando embedding...")
        embedding_usuario = crear_embedding_usuario_nuevo(ratings_onboarding)
        print(f"✅ Embedding creado: shape {embedding_usuario.shape}")
        
        # Generar recomendaciones usando el modelo
        print("🎯 Generando recomendaciones...")
        recomendaciones = generar_recomendaciones(embedding_usuario,generos_seleccionados, 15)
        print(f"✅ Recomendaciones generadas: {len(recomendaciones)}")
        
        # Filtrar películas que ya calificó
        recomendaciones_filtradas = [
            rec for rec in recomendaciones 
            if rec['movieId'] not in ratings_onboarding
        ][:10]
        
        print(f"✅ Recomendaciones filtradas: {len(recomendaciones_filtradas)}")
        
        if not recomendaciones_filtradas:
            generos_seleccionados = request.form.getlist('generos_seleccionados')
            peliculas = obtener_peliculas_onboarding(generos_seleccionados)
            return render_template('onboarding.html', 
                                 peliculas=peliculas,
                                 generos_seleccionados=generos_seleccionados,
                                 error="No se pudieron generar recomendaciones. Intenta calificar películas diferentes.")
        
        return render_template('recomendaciones.html', 
                             recomendaciones=recomendaciones_filtradas,
                             peliculas_calificadas=len(ratings_onboarding))
        
    except Exception as e:
        print(f"💥 Error en procesar_onboarding: {e}")
        import traceback
        traceback.print_exc()
        return render_template('onboarding.html', 
                             peliculas=[],
                             error=f"Error al procesar: {str(e)}")


@recomendador_peliculas_bp.route('/refrescar_peliculas', methods=['POST'])
def refrescar_peliculas():
    """Trae nuevas películas manteniendo las ya calificadas"""
    try:
        # Obtener géneros y películas ya mostradas
        generos_seleccionados = request.form.getlist('generos_seleccionados')
        peliculas_mostradas = request.form.getlist('peliculas_mostradas')
        
        # Obtener calificaciones existentes
        ratings_existentes = {}
        for key, value in request.form.items():
            if key.startswith('rating_') and value:
                movie_id = int(key.replace('rating_', ''))
                ratings_existentes[movie_id] = float(value)
        
        print(f"🔄 Refrescando con {len(ratings_existentes)} películas ya calificadas")
        print(f"🔄 Excluyendo {len(peliculas_mostradas)} películas ya mostradas")
        
        # Obtener nuevas películas excluyendo las ya mostradas
        peliculas_nuevas = obtener_peliculas_onboarding_refresh(
            generos_seleccionados, 
            peliculas_mostradas,
            list(ratings_existentes.keys())
        )
        
        # Renderizar directamente sin usar sesión
        return render_template('onboarding.html', 
                             peliculas=peliculas_nuevas,
                             generos_seleccionados=generos_seleccionados,
                             ratings_existentes=ratings_existentes)
        
    except Exception as e:
        print(f"💥 Error en refrescar_peliculas: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def obtener_peliculas_onboarding_refresh(generos_seleccionados, peliculas_excluir, peliculas_calificadas):
    """Obtiene nuevas películas excluyendo las ya mostradas"""
    try:
        peliculas_excluir_ids = [int(id) for id in peliculas_excluir if id]
        peliculas_seleccionadas = []
        
        # Primero, agregar películas ya calificadas
        for movie_id in peliculas_calificadas:
            pelicula_info = movies[movies['movieId'] == movie_id]
            if len(pelicula_info) > 0:
                movie_row = pelicula_info.iloc[0]
                peliculas_seleccionadas.append({
                    'movieId': int(movie_row['movieId']),
                    'title': movie_row['title'],
                    'genres': movie_row['genres']
                })
        
        # Luego, completar con nuevas películas
        for genero in generos_seleccionados:
            patron = f'(^|\\|){genero}(\\||$)'
            peliculas_del_genero = movies[movies['genres'].str.contains(patron, na=False, regex=True)].copy()
            
            # Excluir películas ya mostradas
            peliculas_del_genero = peliculas_del_genero[~peliculas_del_genero['movieId'].isin(peliculas_excluir_ids)]
            
            if len(peliculas_del_genero) > 0:
                n_peliculas = min(10, len(peliculas_del_genero))
                peliculas_aleatorias = peliculas_del_genero.sample(n=n_peliculas, random_state=None)
                
                for _, movie in peliculas_aleatorias.iterrows():
                    peliculas_seleccionadas.append({
                        'movieId': int(movie['movieId']),
                        'title': movie['title'],
                        'genres': movie['genres']
                    })
        
        print(f"✅ Nuevas películas obtenidas: {len(peliculas_seleccionadas)}")
        return peliculas_seleccionadas
        
    except Exception as e:
        print(f"💥 Error en obtener_peliculas_onboarding_refresh: {e}")
        return []




# Al final del archivo routes.py
print("🚀 Iniciando carga del modelo...")
cargar_modelo_y_datos()
print(f"🔍 Modelo cargado: {modelo is not None}")