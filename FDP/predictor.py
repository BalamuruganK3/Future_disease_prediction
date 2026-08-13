import numpy as np
import pickle, json, os

PREVENTION = {
    'diabetes': {
        'name': 'Diabetes (Sugar Disease)',
        'icon': '🩸',
        'description': 'Your blood sugar level may become too high over time.',
        'prevention': [
            'Cut down on sweets, white rice, maida, and cool drinks',
            'Go for a 30-minute walk every day — even a slow walk helps',
            'Eat smaller meals 4-5 times a day instead of 2 large meals',
            'Drink plain water instead of sugary tea, juice, or cool drinks',
            'Check your sugar level once a year if parents or siblings have diabetes',
            'Avoid eating late at night — finish dinner before 8 PM if possible',
            'Brown rice, ragi, and jowar are better than white rice',
        ],
        'diet': 'Ragi, brown rice, leafy vegetables, sprouts, dal — avoid sweets and fried snacks',
        'exercise': 'Walk 30 minutes every morning or evening, 5 days a week',
        'color': '#e53935'
    },
    'hypertension': {
        'name': 'High Blood Pressure',
        'icon': '💉',
        'description': 'Your blood is pushing too hard against your blood vessels.',
        'prevention': [
            'Eat less salt — avoid pickles, papads, chips, and salty snacks',
            'Eat more bananas, spinach, and coconut water — they help lower BP',
            'Stop or reduce smoking — nicotine raises blood pressure immediately',
            'Limit alcohol to very rare occasions',
            'Walk or exercise 30 minutes daily — 5 days a week',
            'Do deep breathing for 10 minutes before sleeping to relax',
            'Check your BP at a pharmacy or clinic regularly',
        ],
        'diet': 'Less salt, bananas, spinach, coconut water, home-cooked food',
        'exercise': 'Walk 30 minutes every day — swimming or cycling also very good',
        'color': '#e67e22'
    },
    'heart_disease': {
        'name': 'Heart Disease',
        'icon': '❤️',
        'description': 'Your heart or blood vessels may be under strain.',
        'prevention': [
            'Stop smoking immediately — it is the biggest risk for heart problems',
            'Avoid very oily, fried, and junk food — eat home-cooked meals',
            'Control your blood pressure and sugar by checking regularly',
            'Exercise at least 30 minutes daily — even walking counts',
            'Eat fish, walnuts, and use less oil while cooking',
            'Manage stress — long-term tension directly affects the heart',
            'After age 40, get a heart check-up (ECG) at least once a year',
        ],
        'diet': 'Fish, walnuts, less oil, plenty of vegetables, less red meat',
        'exercise': 'Daily 30-minute walk. Avoid very heavy lifting without doctor advice',
        'color': '#c0392b'
    },
    'hypothyroidism': {
        'name': 'Thyroid Problem (Slow Thyroid)',
        'icon': '🔵',
        'description': 'Your thyroid gland in the neck may be working too slowly.',
        'prevention': [
            'Use iodized salt in cooking — most important step for thyroid health',
            'Eat seafood like fish or include milk and curd in your daily diet',
            'Reduce long-term stress — it directly slows the thyroid gland',
            'Sleep 7 to 8 hours every night — poor sleep affects hormones',
            'Get a simple thyroid blood test (TSH) done once a year',
            'If you feel very tired, gaining weight, or always feeling cold — see a doctor',
            'Exercise daily — even a 20-minute walk helps body metabolism',
        ],
        'diet': 'Iodized salt, fish, milk, curd, eggs — reduce raw cabbage and excess soya',
        'exercise': 'Light daily walk or yoga — helps boost energy and body metabolism',
        'color': '#8e44ad'
    },
    'sleep_disorder': {
        'name': 'Sleep Problem',
        'icon': '😴',
        'description': 'You may have trouble falling asleep or feeling rested after sleep.',
        'prevention': [
            'Sleep and wake up at the same time every day including weekends',
            'Keep your bedroom dark, quiet, and slightly cool at night',
            'Put your phone away at least 1 hour before sleeping',
            'Avoid tea or coffee after 3 PM — caffeine stays for many hours',
            'Do not eat heavy food right before sleeping — finish dinner 2 hrs early',
            'Try slow deep breathing or light stretching before bed',
            'Reduce alcohol — it may make you sleep faster but quality is very poor',
        ],
        'diet': 'Warm milk before bed, almonds, banana — avoid heavy dinner or coffee at night',
        'exercise': 'Light evening walk. Avoid gym or running close to bedtime',
        'color': '#2980b9'
    },
    'obesity_risk': {
        'name': 'Weight / Obesity Risk',
        'icon': '⚖️',
        'description': 'Your weight may be too high for your height — this strains heart and joints.',
        'prevention': [
            'Try to eat a little less than usual every day — small changes add up',
            'Eat more vegetables, fruits, and dal — they fill you up with fewer calories',
            'Switch from white rice and maida to whole wheat, ragi, or millets',
            'Stop or cut down on cool drinks, juice packets, and sweets completely',
            'Drink 8 to 10 glasses of water daily — thirst often feels like hunger',
            'Walk at least 30 to 45 minutes every day — no gym needed to start',
            'Do not eat while watching TV or phone — you eat much more without noticing',
        ],
        'diet': 'Vegetables, dal, eggs, small portions of rice or roti — cut junk food',
        'exercise': '45-minute daily walk + simple exercises like squats or jumping jacks',
        'color': '#27ae60'
    },
    'occupational_disease': {
        'name': 'Work-Related Health Risk',
        'icon': '🏭',
        'description': 'Your job type puts you at risk from physical strain, dust, or chemicals.',
        'prevention': [
            'Always wear protective equipment — mask, gloves, helmet as needed at work',
            'Take short breaks every 1 hour — stand up, stretch, and walk a little',
            'If you sit at a desk, keep your chair and screen at the right height',
            'Get a full health check-up once a year — work diseases can be caught early',
            'If you notice unusual pain, breathlessness, or skin problems — report it',
            'Exercise daily to recover from the physical strain of your job',
            'Drink enough water, especially if your job involves heat or outdoor work',
        ],
        'diet': 'Fresh fruits, vegetables daily, avoid heavy oily meals during work hours',
        'exercise': 'Stretching breaks every hour at work. Light walk or yoga after work',
        'color': '#f39c12'
    },
    'respiratory_disease': {
        'name': 'Respiratory Disease (Asthma / COPD)',
        'icon': '🫁',
        'description': 'Your lungs may be getting affected by air pollution, dust, or smoke.',
        'prevention': [
            'Check your city AQI every morning — stay indoors on days above 150 AQI',
            'Wear an N95 mask when going outside in polluted or dusty areas',
            'Never smoke and avoid being near people who are smoking',
            'Keep your home well ventilated — open windows when outdoor air is clean',
            'Avoid burning wood, garbage, or leaves near your home',
            'If you work in factory or dusty areas, always wear a proper dust mask',
            'Do breathing exercises like pranayama or pursed lip breathing daily',
            'Visit a doctor for a lung function test if you feel breathless often',
        ],
        'diet': 'Turmeric milk, ginger tea, honey, amla, and citrus fruits like orange and guava',
        'exercise': 'Light breathing exercises and yoga. Avoid outdoor running on high AQI days',
        'color': '#00838f'
    },
}


def load_model_and_scaler():
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, 'models', 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(base, 'models', 'features.json')) as f:
        meta = json.load(f)
    mpath = os.path.join(base, 'models', 'sklearn_model.pkl')
    with open(mpath, 'rb') as f:
        model = pickle.load(f)
    meta['backend'] = 'sklearn'
    return model, scaler, meta


def predict(user_input, model, scaler, meta):
    bmi_val = round(user_input['weight'] / ((user_input['height'] / 100) ** 2), 2)
    user_input['bmi'] = bmi_val
    if 'aqi_category' not in user_input:
        user_input['aqi_category'] = 1

    features = meta['features']
    diseases = meta['diseases']

    X = np.array([[user_input[f] for f in features]], dtype=float)
    X_scaled = scaler.transform(X)

    probs = []
    for est in model.estimators_:
        p = est.predict_proba(X_scaled)[0]
        probs.append(p[1] if len(p) > 1 else float(est.predict(X_scaled)[0]))
    probs = np.array(probs)

    results = []
    for i, disease in enumerate(diseases):
        prob = float(probs[i])
        info = PREVENTION[disease]
        risk = 'HIGH' if prob >= 0.6 else 'MEDIUM' if prob >= 0.35 else 'LOW'
        results.append({
            'disease':     disease,
            'name':        info['name'],
            'icon':        info['icon'],
            'probability': round(prob * 100, 1),
            'risk':        risk,
            'description': info['description'],
            'prevention':  info['prevention'],
            'diet':        info['diet'],
            'exercise':    info['exercise'],
            'color':       info['color'],
        })
    results.sort(key=lambda x: x['probability'], reverse=True)
    return {'bmi': bmi_val, 'predictions': results}
