from flask import render_template, redirect, request, session, flash, jsonify
import os
from . import visitas_log_bp
from .visit_logger import get_all_visits, delete_visit

@visitas_log_bp.route('/visitas', methods=['GET', 'POST'])
def visitas():
    """Página de visitas con autenticación por contraseña"""
    if request.method == 'POST':
        password = request.form.get('password')
        correct_password = os.getenv('VISITAS_PASSWORD', 'admin123')
        
        if password == correct_password:
            session['visitas_authenticated'] = True
            return redirect('/visitas')
        else:
            flash('Contraseña incorrecta', 'error')
    
    # Verificar autenticación
    if not session.get('visitas_authenticated', False):
        return render_template('visitas_login.html')
    
    # Obtener todas las visitas
    visits = get_all_visits()
    return render_template('visitas.html', visits=visits)

@visitas_log_bp.route('/visitas/logout')
def visitas_logout():
    """Cerrar sesión de visitas"""
    session.pop('visitas_authenticated', None)
    return redirect('/')

@visitas_log_bp.route('/visitas/delete/<int:visit_id>', methods=['POST'])
def delete_visit_route(visit_id):
    """Eliminar una visita específica"""
    # Verificar autenticación
    if not session.get('visitas_authenticated', False):
        return jsonify({'success': False, 'message': 'No autorizado'}), 401
    
    success = delete_visit(visit_id)
    if success:
        return jsonify({'success': True, 'message': 'Visita eliminada correctamente'})
    else:
        return jsonify({'success': False, 'message': 'Error al eliminar la visita'}), 500