from flask import Flask, request, jsonify, render_template
import os
import pickle
import json

from run_pipeline import train_pipeline


def create_app():
    app = Flask(__name__)

    model_path = "customer_spending_model.pkl"
    scaler_path = "scaler.pkl"
    features_path = "features.json"

    # Ensure model and scaler exist
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
        # train and save
        train_pipeline()

    pipeline = None
    pipeline_path = "pipeline.pkl"
    if os.path.exists(pipeline_path):
        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)

    if pipeline is None:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        # keep model/scaler references for backward compatibility
        model = pipeline.named_steps.get('model')
        scaler = pipeline.named_steps.get('scaler')
    with open(features_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    features = meta.get("features")


    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html", features=features)


    @app.route("/predict", methods=["POST"])
    def predict():
        # accept JSON or form data
        data = request.get_json(silent=True)
        if data is None:
            data = request.form.to_dict()

        # normalize incoming keys to simple lower_snake
        data_map = {}
        if isinstance(data, dict):
            for k, v in data.items():
                nk = k.strip().lower().replace(' ', '_')
                data_map[nk] = v

        # compute derived features if possible
        try:
            # compute Total_Items from base parts if missing
            if 'total_items' not in data_map:
                parts = [data_map.get('apparel_items'), data_map.get('fnv_items'), data_map.get('staples_items')]
                if all(p is not None for p in parts):
                    data_map['total_items'] = float(parts[0]) + float(parts[1]) + float(parts[2])

            # compute Items_Per_Visit if missing
            if 'items_per_visit' not in data_map:
                tiv = None
                if 'total_items' in data_map and 'no_of_visits' in data_map:
                    visits = float(data_map['no_of_visits'])
                    if visits != 0:
                        tiv = float(data_map['total_items']) / visits
                    else:
                        tiv = 0.0
                if tiv is not None:
                    data_map['items_per_visit'] = tiv
        except Exception:
            return jsonify({"error": "Failed to compute derived features"}), 400

        # build feature vector in the same order
        vals = []
        missing = []
        for feat in features:
            key = feat.strip()
            nk = key.lower().replace(' ', '_')
            if nk not in data_map:
                missing.append(key)
                continue
            try:
                vals.append(float(data_map[nk]))
            except Exception:
                return jsonify({"error": f"Invalid numeric value for {key}: {data_map[nk]}"}), 400

        if missing:
            return jsonify({"error": "Missing features", "missing": missing}), 400

        try:
            import numpy as np
            X = np.array(vals).reshape(1, -1)
            if pipeline is not None:
                pred = pipeline.predict(X)
            else:
                Xs = scaler.transform(X)
                pred = model.predict(Xs)
            return jsonify({"prediction": float(pred[0])})
        except Exception as e:
            app.logger.exception('Prediction failure')
            return jsonify({"error": "Prediction failed", "details": str(e)}), 500

    @app.route('/info', methods=['GET'])
    def info():
        info = {
            'features': features,
            'model_path': model_path,
        }
        if os.path.exists('metrics.json'):
            try:
                with open('metrics.json', 'r', encoding='utf-8') as f:
                    info['metrics'] = json.load(f)
            except Exception:
                info['metrics'] = None
        return jsonify(info)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)
