from flask import Flask, render_template
import os

# Importar los blueprints
from predictor_sueldos import predictor_sueldos_bp

# Crear la aplicación Flask
app = Flask(__name__)

# Registrar los blueprints
app.register_blueprint(predictor_sueldos_bp)

@app.route('/')
def home():
    """Página principal con navegación a los diferentes proyectos"""
    return render_template('home.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)