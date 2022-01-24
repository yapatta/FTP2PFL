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
    cors  = CORS(app)

    @app.route("/clients/<client_id>", methods=['POST'])
    @cross_origin(origin="*")
    def clients(client_id):
        dumped_json = json.dumps(request.json)
        model = (json.loads(dumped_json))["model"]
        bmodel = base64.b64decode((model.encode()))

        client_id = int(client_id)
        weights, n, loss, acc = client.learn(client_id, bmodel)
        buf = ag.byte_weights(weights)
        return jsonify({'model': base64.b64encode(buf).decode('utf-8'), 'num': n, 'loss': loss, 'acc': acc}), 200

    @app.route("/initial", methods=['GET'])
    @cross_origin(origin="*")
    def initial():
        weights = ag.initial_learn()
        buf = ag.byte_weights(weights)
        return jsonify({'model': base64.b64encode(buf).decode('utf-8')}), 200

    @app.route("/aggregate/<leader_id>", methods=['POST'])
    @cross_origin(origin="*")
    def aggregate(leader_id):
        dumped_json = json.dumps(request.json)
        # バイト列のList
        bmodels_num = [ (base64.b64decode((params["model"].encode())), int(params["num"])) for params in (json.loads(dumped_json))["models"]]
        bmodels = [x[0] for x in bmodels_num]
        nums = [x[1] for x in bmodels_num]

        leader_id = int(leader_id)
        weights, loss, acc = ag.aggregate(leader_id, bmodels, nums)
        buf = ag.byte_weights(weights)
        return jsonify({'model': base64.b64encode(buf).decode('utf-8'), 'loss': loss, 'acc': acc}), 200

    return app


serve(create_app(), host='0.0.0.0', port=9000, threads=12)
