from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/crop_recommendation')
def crop_recommendation():
    # Your recommendation logic here
    return jsonify([
        {"crop": "pomegranate", "probability": 0.5796053652207233},
        {"crop": "jute", "probability": 0.12387259898754807},
        {"crop": "mango", "probability": 0.03139580365281054},
        {"crop": "watermelon", "probability": 0.026992217422372365},
        {"crop": "coconut", "probability": 0.022128156697916384}
    ])

if __name__ == '__main__':
    app.run(port=5000)
