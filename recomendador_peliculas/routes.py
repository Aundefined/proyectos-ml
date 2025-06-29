from flask import Blueprint, render_template, request, jsonify
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
ratings = None
movies = None
user_to_index = None
movie_to_index = None
user_embeddings = None
movie_embeddings = None

def cargar_modelo_y_datos():
    """Carga el modelo entrenado y los datos necesarios"""
    global modelo, ratings, movies, user_to_index, movie_to_index, user_embeddings, movie_embeddings
    
    try:
        print("🔄 Intentando cargar modelo...")
        
        # Cargar modelo
        modelo = tf.keras.models.load_model('ml-models/modelo_recomendacion.h5', compile=False)
        print("✓ Modelo cargado sin compilar")
        
        # Recompilar manualmente
        modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print("✓ Modelo recompilado")
        
        # Cargar datos
        ratings = pd.read_csv('ml-models/ratings.csv')
        movies = pd.read_csv('ml-models/movies.csv')
        print("✓ CSVs cargados")
        
        # Cargar mapeos
        with open('ml-models/mapeos_recomendacion.pkl', 'rb') as f:
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
    
def obtener_peliculas_onboarding():
    """Obtiene las películas más populares para el onboarding"""
    try:
        # Verificar que tenemos datos
        if ratings is None or movies is None:
            raise Exception("Datos no disponibles")
        
        print(f"📊 Ratings shape: {ratings.shape}")
        print(f"📊 Movies shape: {movies.shape}")
        
        # Películas más populares (con más ratings)
        popular_movies = (ratings.groupby('movieId')
                         .size()
                         .sort_values(ascending=False)
                         .head(30)
                         .index.tolist())
        
        print(f"📊 Popular movies: {len(popular_movies)}")
        
        # Unir con datos de películas para obtener títulos
        peliculas_onboarding = movies[movies['movieId'].isin(popular_movies)]
        
        print(f"📊 Películas encontradas: {len(peliculas_onboarding)}")
        
        # Seleccionar las primeras 15 (simplificado)
        peliculas_seleccionadas = []
        for _, movie in peliculas_onboarding.head(15).iterrows():
            peliculas_seleccionadas.append({
                'movieId': int(movie['movieId']),
                'title': movie['title'],
                'genres': movie['genres']
            })
        
        print(f"📊 Películas seleccionadas: {len(peliculas_seleccionadas)}")
        return peliculas_seleccionadas
        
    except Exception as e:
        print(f"💥 Error en obtener_peliculas_onboarding: {e}")
        import traceback
        traceback.print_exc()
        return []

def calcular_correlacion(user_id, ratings_onboarding):
    """Calcula correlación entre un usuario del dataset y el usuario nuevo"""
    try:
        # Obtener ratings del usuario existente
        user_ratings = ratings[ratings['userId'] == user_id]
        
        # Encontrar películas en común
        peliculas_comunes = set(user_ratings['movieId']).intersection(set(ratings_onboarding.keys()))
        
        # Si hay menos de 3 películas en común, retornar 0
        if len(peliculas_comunes) < 3:
            return 0.0
        
        # Extraer ratings para películas comunes
        ratings_usuario_existente = []
        ratings_usuario_nuevo = []
        
        for movie_id in peliculas_comunes:
            # Rating del usuario existente
            rating_existente = user_ratings[user_ratings['movieId'] == movie_id]['rating'].iloc[0]
            ratings_usuario_existente.append(rating_existente)
            
            # Rating del usuario nuevo
            ratings_usuario_nuevo.append(ratings_onboarding[movie_id])
        
        # Verificar que hay variación en los datos
        if len(set(ratings_usuario_existente)) <= 1 or len(set(ratings_usuario_nuevo)) <= 1:
            return 0.0  # No hay variación, correlación no definida
        
        # Calcular correlación de Pearson
        if len(ratings_usuario_existente) < 2:
            return 0.0
        
        correlacion, _ = pearsonr(ratings_usuario_existente, ratings_usuario_nuevo)
        
        # Manejar NaN y valores inválidos
        if np.isnan(correlacion) or np.isinf(correlacion):
            return 0.0
        
        return max(0.0, correlacion)  # Solo correlaciones positivas
        
    except Exception as e:
        print(f"Error en calcular_correlacion para usuario {user_id}: {e}")
        return 0.0

def crear_embedding_usuario_nuevo(ratings_onboarding):
    """Crea un embedding para un usuario nuevo basado en usuarios similares"""
    try:
        print(f"🔍 user_embeddings shape: {user_embeddings.shape if user_embeddings is not None else 'None'}")
        print(f"🔍 user_to_index keys: {len(user_to_index) if user_to_index else 0}")
        
        if user_embeddings is None:
            raise Exception("user_embeddings es None")
        
        usuarios_similares = []
        correlaciones = []
        
        print(f"🔍 Buscando usuarios similares para {len(ratings_onboarding)} ratings...")
        
        # Buscar usuarios similares
        for user_id in range(1, min(100, 611)):  # Probar solo los primeros 100 usuarios
            correlacion = calcular_correlacion(user_id, ratings_onboarding)
            if correlacion > 0.2:
                usuarios_similares.append(user_id)
                correlaciones.append(correlacion)
        
        print(f"🔍 Usuarios similares encontrados: {len(usuarios_similares)}")
        
        if not usuarios_similares:
            print("⚠️ No se encontraron usuarios similares, usando promedio general")
            return np.mean(user_embeddings, axis=0)
        
        # Crear embedding promediando usuarios similares
        embeddings_similares = []
        for user_id in usuarios_similares[:10]:  # Solo los primeros 10
            if user_id in user_to_index:
                user_idx = user_to_index[user_id]
                if user_idx < len(user_embeddings):  # Verificar bounds
                    embeddings_similares.append(user_embeddings[user_idx])
        
        if not embeddings_similares:
            print("⚠️ No se pudieron obtener embeddings, usando promedio general")
            return np.mean(user_embeddings, axis=0)
        
        # Promedio simple
        embedding_nuevo = np.mean(embeddings_similares, axis=0)
        print(f"✅ Embedding creado con {len(embeddings_similares)} usuarios")
        
        return embedding_nuevo
        
    except Exception as e:
        print(f"💥 Error en crear_embedding_usuario_nuevo: {e}")
        # Fallback: usar embedding del usuario 0
        return user_embeddings[0] if user_embeddings is not None else np.zeros(50)

def generar_recomendaciones(embedding_usuario, n_recomendaciones=10):
   """Genera recomendaciones usando el modelo entrenado con predicciones reales"""
   try:
       print(f"🎯 Generando recomendaciones reales...")
       
       # Crear el modelo de embeddings directos (una sola vez)
       modelo_embeddings = crear_modelo_embeddings_directos()

       
       predicciones = []
       peliculas_disponibles = movies.head(200)
       
       for _, movie_row in peliculas_disponibles.iterrows():
           try:
               movie_id = movie_row['movieId']
               
               if movie_id not in movie_to_index:
                   continue
                   
               movie_idx = movie_to_index[movie_id]
               if movie_idx >= len(movie_embeddings):
                   continue
               
               movie_embedding = movie_embeddings[movie_idx]
               
               # Usar el modelo para predecir
               prediction = modelo_embeddings.predict([
                   embedding_usuario.reshape(1,-1), 
                   movie_embedding.reshape(1,-1)
               ])[0][0]
               
               predicted_rating = max(1.0, min(5.0, prediction))
               
               predicciones.append({
                   'movieId': int(movie_id),
                   'predicted_rating': float(predicted_rating),
                   'title': movie_row['title'],
                   'genres': movie_row['genres']
               })
               
           except:
               continue
       
       print(f"✅ Total predicciones calculadas: {len(predicciones)}")
       
      
       
       # Ordenar por rating predicho
       predicciones.sort(key=lambda x: x['predicted_rating'], reverse=True)
       
       return predicciones[:n_recomendaciones]
       
   except Exception as e:
       print(f"💥 Error en generar_recomendaciones: {e}")
     
     
     
def crear_modelo_embeddings_directos():
    """Crea un modelo que acepta embeddings directos"""
    from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
    from tensorflow.keras.models import Model
    
    # Inputs directos de embeddings (no índices)
    user_embedding_input = Input(shape=(50,), name='user_embedding')
    movie_embedding_input = Input(shape=(50,), name='movie_embedding')
    
    # Concatenar (igual que tu modelo original)
    concat = Concatenate()([user_embedding_input, movie_embedding_input])
    
    # Capas densas (copiar tu arquitectura)
    dense1 = Dense(16, activation='relu')(concat)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(16, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    output = Dense(1, activation='linear')(dropout2)
    
    modelo_embeddings = Model(inputs=[user_embedding_input, movie_embedding_input], outputs=output)
    
    # Copiar los pesos SOLO de las capas que tienen pesos
    modelo_embeddings.layers[3].set_weights(modelo.layers[7].get_weights())  # dense1 (índice 3 en nuevo modelo)
    modelo_embeddings.layers[5].set_weights(modelo.layers[9].get_weights())  # dense2 (índice 5 en nuevo modelo)
    modelo_embeddings.layers[7].set_weights(modelo.layers[11].get_weights()) # output (índice 7 en nuevo modelo)
    
    return modelo_embeddings


@recomendador_peliculas_bp.route('/')
def index():
    """Página principal del recomendador"""
    return render_template('recomendador_peliculas.html')


@recomendador_peliculas_bp.route('/onboarding')
def onboarding():
    global modelo, ratings, movies
    
    # LLAMAR AQUÍ DIRECTAMENTE
    if modelo is None:
        print("Modelo es None, cargando...")
        cargar_modelo_y_datos()
    
    if not modelo:
        return "<h1>Error: No se pudo cargar el modelo</h1>"
    
    peliculas = obtener_peliculas_onboarding()
    return render_template('onboarding.html', peliculas=peliculas)


@recomendador_peliculas_bp.route('/procesar_onboarding', methods=['POST'])
def procesar_onboarding():
    """Procesa las calificaciones del onboarding y genera recomendaciones"""
    try:
        print("🚀 Iniciando procesamiento...")
        
        # Obtener ratings del formulario
        ratings_onboarding = {}
        for key, value in request.form.items():
            if key.startswith('rating_') and value:
                movie_id = int(key.replace('rating_', ''))
                ratings_onboarding[movie_id] = float(value)
        
        print(f"📝 Ratings recibidos: {ratings_onboarding}")
        
        # Validar que se calificaron al menos 5 películas
        if len(ratings_onboarding) < 5:
            return render_template('onboarding.html', 
                                 peliculas=obtener_peliculas_onboarding(),
                                 error="Por favor, califica al menos 5 películas")
        
        print("✅ Validación pasada")
        
        # Crear embedding para usuario nuevo
        print("🧠 Creando embedding...")
        embedding_usuario = crear_embedding_usuario_nuevo(ratings_onboarding)
        print(f"✅ Embedding creado: shape {embedding_usuario.shape}")
        
        # Generar recomendaciones usando el modelo
        print("🎯 Generando recomendaciones...")
        recomendaciones = generar_recomendaciones(embedding_usuario, 15)
        print(f"✅ Recomendaciones generadas: {len(recomendaciones)}")
        
        # Filtrar películas que ya calificó
        recomendaciones_filtradas = [
            rec for rec in recomendaciones 
            if rec['movieId'] not in ratings_onboarding
        ][:10]
        
        print(f"✅ Recomendaciones filtradas: {len(recomendaciones_filtradas)}")
        
        if not recomendaciones_filtradas:
            return render_template('onboarding.html', 
                                 peliculas=obtener_peliculas_onboarding(),
                                 error="No se pudieron generar recomendaciones. Intenta calificar películas diferentes.")
        
        return render_template('recomendaciones.html', 
                             recomendaciones=recomendaciones_filtradas,
                             peliculas_calificadas=len(ratings_onboarding))
        
    except Exception as e:
        print(f"💥 Error en procesar_onboarding: {e}")
        import traceback
        traceback.print_exc()
        return render_template('onboarding.html', 
                             peliculas=obtener_peliculas_onboarding(),
                             error=f"Error al procesar: {str(e)}")



# Al final del archivo routes.py
print("🚀 Iniciando carga del modelo...")
cargar_modelo_y_datos()
print(f"🔍 Modelo cargado: {modelo is not None}")