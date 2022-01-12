from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from waitress import serve
import base64
import os
import json

import aggregator as ag
import client


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    cors = CORS(app)

    @app.route("/clients/<client_id>", methods=['GET'])
    @cross_origin(origin="*")
    def clients(client_id):
        client_id = int(client_id)
        weights = client.learn(client_id)
        buf = ag.byte_weights(weights)
        return jsonify({'model': base64.b64encode(buf).decode('utf-8')}), 200

    @app.route("/initial", methods=['GET'])
    @cross_origin(origin="*")
    def initial():
        weights = ag.initial_learn() if not os.path.exists(
            ag.MODEL_FILE) else ag.load_parent_model()
        buf = ag.byte_weights(weights)
        return jsonify({'model': base64.b64encode(buf).decode('utf-8')}), 200

    @app.route("/aggregate", methods=['POST'])
    @cross_origin(origin="*")
    def aggregate():
        dumped_json = json.dumps(request.json)
        # バイト列のList
        bmodels = map(lambda m: base64.b64decode(
            (m.encode())), (json.loads(dumped_json))["models"])
        weights = ag.aggregate(bmodels)
        buf = ag.byte_weights(weights)
        return jsonify({'model': base64.b64encode(buf).decode('utf-8')}), 200

    return app


serve(create_app(), host='0.0.0.0', port=9000, threads=10)
