from flask import Blueprint

chatbot_blaniza_bp = Blueprint('chatbot_blaniza', __name__, 
    url_prefix='/chatbot-blaniza',
    template_folder='templates',
    static_folder='static',
    static_url_path='/static/chatbot_blaniza'
)

from . import routes