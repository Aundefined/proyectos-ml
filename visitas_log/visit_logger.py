import sqlite3
import os
import requests
from datetime import datetime
from flask import request

DATABASE_PATH = 'visits.db'

def init_db():
    """Inicializar la base de datos y crear la tabla de visitas si no existe"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS visitas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            FECHA_HORA DATETIME NOT NULL,
            URL TEXT NOT NULL,
            DIRECCION_IP TEXT NOT NULL,
            country TEXT,
            regionName TEXT,
            city TEXT,
            zip TEXT,
            lat REAL,
            lon REAL
        )
    ''')
    
    # Agregar las nuevas columnas si la tabla ya existe (migración)
    try:
        cursor.execute('ALTER TABLE visitas ADD COLUMN country TEXT')
    except sqlite3.OperationalError:
        pass  # La columna ya existe
    
    try:
        cursor.execute('ALTER TABLE visitas ADD COLUMN regionName TEXT')
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute('ALTER TABLE visitas ADD COLUMN city TEXT')
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute('ALTER TABLE visitas ADD COLUMN zip TEXT')
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute('ALTER TABLE visitas ADD COLUMN lat REAL')
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute('ALTER TABLE visitas ADD COLUMN lon REAL')
    except sqlite3.OperationalError:
        pass
    
    conn.commit()
    conn.close()

def log_visit(url, ip_address):
    """Registrar una visita en la base de datos"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        fecha_hora = datetime.now()
        
        # Obtener datos de geolocalización
        geo_data = get_geo_data(ip_address)
        
        if geo_data:
            cursor.execute('''
                INSERT INTO visitas (FECHA_HORA, URL, DIRECCION_IP, country, regionName, city, zip, lat, lon)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (fecha_hora, url, ip_address, 
                  geo_data['country'], geo_data['regionName'], geo_data['city'], 
                  geo_data['zip'], geo_data['lat'], geo_data['lon']))
        else:
            cursor.execute('''
                INSERT INTO visitas (FECHA_HORA, URL, DIRECCION_IP)
                VALUES (?, ?, ?)
            ''', (fecha_hora, url, ip_address))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error al registrar visita: {e}")

def get_all_visits():
    """Obtener todas las visitas ordenadas por fecha descendente"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, FECHA_HORA, URL, DIRECCION_IP, country, regionName, city, zip, lat, lon
            FROM visitas
            ORDER BY FECHA_HORA DESC
        ''')
        
        visits = cursor.fetchall()
        conn.close()
        
        # Formatear fechas
        formatted_visits = []
        for visit in visits:
            visit_id = visit[0]
            fecha_str = visit[1]
            # Convertir string de fecha a datetime
            fecha_dt = datetime.strptime(fecha_str, '%Y-%m-%d %H:%M:%S.%f')
            # Formatear a dd/MM/yyyy-hh:mm
            fecha_formateada = fecha_dt.strftime('%d/%m/%Y-%H:%M')
            
            # Extraer sección de la URL
            full_url = visit[2]
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(full_url)
                path = parsed_url.path.strip('/')
                section = path if path else 'home'
            except:
                section = 'home'
            
            # Crear enlace a Google Maps si hay coordenadas
            maps_link = None
            if visit[8] is not None and visit[9] is not None:  # lat y lon
                maps_link = f"https://www.google.com/maps?q={visit[8]},{visit[9]}"
            
            formatted_visits.append((
                visit_id,           # 0: id
                fecha_formateada,   # 1: fecha
                full_url,          # 2: url completa (para el enlace)
                section,           # 3: sección (para mostrar)
                visit[3],          # 4: ip (de BD)
                visit[4],          # 5: country (de BD)
                visit[5],          # 6: regionName (de BD)
                visit[6],          # 7: city (de BD)
                visit[7],          # 8: zip (de BD)
                maps_link          # 9: enlace a Google Maps
            ))
        
        return formatted_visits
    except Exception as e:
        print(f"Error al obtener visitas: {e}")
        return []

def delete_visit(visit_id):
    """Eliminar una visita de la base de datos"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM visitas WHERE id = ?
        ''', (visit_id,))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error al eliminar visita: {e}")
        return False

def get_geo_data(ip_address):
    """Obtener datos de geolocalización para una IP"""
    if ip_address == '127.0.0.1' or ip_address == 'localhost':
        return None
    
    try:
        response = requests.get(f'http://ip-api.com/json/{ip_address}', timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                return {
                    'country': data.get('country'),
                    'regionName': data.get('regionName'),
                    'city': data.get('city'),
                    'zip': data.get('zip'),
                    'lat': data.get('lat'),
                    'lon': data.get('lon')
                }
    except Exception as e:
        print(f"Error al obtener datos de geolocalización: {e}")
    
    return None

def get_client_ip():
    """Obtener la dirección IP real del cliente"""
    if request.headers.getlist("X-Forwarded-For"):
        ip = request.headers.getlist("X-Forwarded-For")[0].split(',')[0]
    else:
        ip = request.remote_addr
    return ip