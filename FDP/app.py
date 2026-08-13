from flask import Flask, render_template, request, jsonify, make_response
import os, json, sys, traceback, threading

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# ── Pre-load model in background thread at startup ─────────────
_model = _scaler = _meta = None
_ready = False
_load_error = None

def _preload():
    global _model, _scaler, _meta, _ready, _load_error
    try:
        from predictor import load_model_and_scaler
        _model, _scaler, _meta = load_model_and_scaler()
        _ready = True
        print("  [OK] Model loaded and ready!")
    except Exception as e:
        _load_error = str(e)
        print(f"  [ERROR] Model load failed: {e}")

# Start loading immediately when app.py is imported
_thread = threading.Thread(target=_preload, daemon=True)
_thread.start()

JOB_LIST = [
    "Software Engineer / IT","Doctor","Teacher","Farmer",
    "Driver (Auto / Cab / Lorry)","Nurse","Office Worker",
    "Construction Worker","Chef / Cook","Student",
    "Manager / Supervisor","Retired / Senior Citizen","Businessman",
    "Security Guard / Watchman","Factory Worker",
    "Homemaker / Housewife","House Maid / Domestic Worker",
    "Delivery Boy / Courier","Auto / Cab Driver","Shopkeeper / Vendor",
    "Tailor","Electrician / Plumber","Mechanic",
    "Daily Wage Worker","Street Vendor",
]
BLOOD_GROUPS = ["A+","A-","B+","B-","AB+","AB-","O+","O-"]
AQI_LABELS = [
    "Good (AQI 0-50) - Clean air",
    "Moderate (AQI 51-100) - Acceptable",
    "Unhealthy for sensitive (AQI 101-150)",
    "Unhealthy (AQI 151-200) - Everyone affected",
    "Very unhealthy (AQI 201-300) - Health alert",
    "Hazardous (AQI 301+) - Emergency",
]


@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return r


@app.route("/")
def index():
    return render_template("index.html",
        jobs=enumerate(JOB_LIST),
        blood_groups=enumerate(BLOOD_GROUPS),
        aqi_labels=enumerate(AQI_LABELS))


@app.route("/ready")
def ready():
    """Frontend polls this until model is loaded."""
    if _ready:
        return jsonify({"ready": True})
    if _load_error:
        return jsonify({"ready": False, "error": _load_error})
    return jsonify({"ready": False})


@app.route("/predict", methods=["POST","OPTIONS"])
def predict_route():
    if request.method == "OPTIONS":
        return make_response("", 200)

    if not _ready:
        return jsonify({"success": False,
            "error": "Model is still loading. Please wait a few seconds and try again."}), 503

    try:
        data = request.get_json(force=True, silent=True) or request.form.to_dict()
        if not data:
            return jsonify({"success": False, "error": "No data received"}), 400

        def gi(k, d=0):
            try:    return int(float(data.get(k, d)))
            except: return d
        def gf(k, d=0.0):
            try:    return float(data.get(k, d))
            except: return d

        inp = {
            "age":                    gi("age", 30),
            "weight":                 gf("weight", 70),
            "height":                 gf("height", 170),
            "blood_group":            gi("blood_group", 0),
            "job":                    gi("job", 0),
            "sleep_hours":            gf("sleep_hours", 7),
            "stress_level":           gi("stress_level", 5),
            "physical_activity_days": gi("physical_activity_days", 3),
            "smoking":                gi("smoking", 0),
            "alcohol":                gi("alcohol", 0),
            "family_history_diabetes":gi("family_history_diabetes", 0),
            "family_history_heart":   gi("family_history_heart", 0),
            "systolic_bp":            gi("systolic_bp", 120),
            "diastolic_bp":           gi("diastolic_bp", 80),
            "fasting_glucose":        gi("fasting_glucose", 95),
            "aqi_category":           gi("aqi_category", 1),
        }

        errs = []
        if not (1 <= inp["age"] <= 120):       errs.append(f"Age must be 1-120")
        if inp["sleep_hours"] <= 0:             errs.append("Sleep cannot be 0 - minimum is 1 hour")
        elif inp["sleep_hours"] > 24:           errs.append("Sleep cannot exceed 24 hours")
        else: inp["sleep_hours"] = max(1.0, min(float(inp["sleep_hours"]), 10.0))
        if inp["stress_level"] <= 0:            errs.append("Stress level cannot be 0 - minimum is 1")
        else: inp["stress_level"] = max(1, min(inp["stress_level"], 10))
        if not (0 <= inp["physical_activity_days"] <= 7): errs.append("Exercise days must be 0-7")
        if not (10 <= inp["weight"] <= 300):    errs.append("Weight must be 10-300 kg")
        if not (50 <= inp["height"] <= 250):    errs.append("Height must be 50-250 cm")
        if not (60 <= inp["systolic_bp"] <= 250):  errs.append("Upper BP must be 60-250")
        if not (40 <= inp["diastolic_bp"] <= 150): errs.append("Lower BP must be 40-150")
        if not (40 <= inp["fasting_glucose"] <= 600): errs.append("Blood sugar must be 40-600")
        inp["aqi_category"] = max(0, min(inp["aqi_category"], 5))

        if errs:
            return jsonify({"success": False, "error": " | ".join(errs)}), 400

        from predictor import predict as do_predict
        result = do_predict(inp, _model, _scaler, _meta)

        job_idx = max(0, min(inp["job"], len(JOB_LIST)-1))
        bg_idx  = max(0, min(inp["blood_group"], len(BLOOD_GROUPS)-1))
        result["job_label"]         = JOB_LIST[job_idx]
        result["blood_group_label"] = BLOOD_GROUPS[bg_idx]
        result["age"]               = inp["age"]

        return jsonify({"success": True, "result": result})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/accuracy")
def accuracy():
    try:
        with open(os.path.join(BASE, "models", "accuracy_results.json")) as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    print("=" * 55)
    print("  Future Disease Prediction System")
    print("  Model loading in background...")
    print("  Open: http://localhost:5000")
    print("  The page will show a loading bar until ready")
    print("=" * 55)
    app.run(debug=False, port=5000, host="0.0.0.0",
            use_reloader=False, threaded=True)
