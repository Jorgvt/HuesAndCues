import os
import sys

# Ensure project root and AppVisualizer are in sys.path
APP_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from flask import Flask, render_template, jsonify, request

try:
    from AppVisualizer.data_loader import DataLoader
except ModuleNotFoundError:
    from data_loader import DataLoader

app = Flask(__name__, template_folder='templates', static_folder='static')

# Initialize Data Loader
data_loader = DataLoader()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/board')
def get_board():
    return jsonify(data_loader.get_board_grid())

@app.route('/api/words')
def get_words():
    only_cb = request.args.get('only_cb', 'false').lower() == 'true'
    words = data_loader.get_words_list()
    if only_cb:
        words = [w for w in words if w['has_colorblind']]
    return jsonify(words)

@app.route('/api/word/<word_name>')
def get_word(word_name):
    analysis = data_loader.get_word_analysis(word_name)
    if analysis is None:
        return jsonify({"error": "Word not found"}), 404
    return jsonify(analysis)

@app.route('/api/colorblind_users')
def get_colorblind_users():
    return jsonify(data_loader.get_colorblind_summary())

if __name__ == '__main__':
    import socket
    
    def find_free_port(default_port=5000):
        for p in [default_port, 5001, 8080, 8000]:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', p)) != 0:
                    return p
        return default_port

    port = int(os.environ.get('PORT', find_free_port(5000)))
    print(f"Starting Hues & Cues App Visualizer server at http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)

