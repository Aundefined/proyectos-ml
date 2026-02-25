from flask import Blueprint

chatbot_gemelo_bp = Blueprint('chatbot_gemelo', __name__,
    url_prefix='/chatbot-gemelo',
    template_folder='templates',
    static_folder='static',
    static_url_path='/static/chatbot_gemelo'
)

from . import routes
