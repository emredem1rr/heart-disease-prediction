from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Model dosyalarını yükle
print("📦 Model dosyaları yükleniyor...")
model = pickle.load(open("uci_model.pkl", "rb"))
scaler = pickle.load(open("uci_scaler.pkl", "rb"))
imputer = pickle.load(open("uci_imputer.pkl", "rb"))
features = pickle.load(open("uci_features.pkl", "rb"))
metrics = pickle.load(open("uci_metrics.pkl", "rb")) if os.path.exists("uci_metrics.pkl") else None

print("✅ Model yüklendi!")
print(f"📊 Beklenen özellikler: {features}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Form verilerini al
        user_data = {
            'age': float(request.form.get('age')),
            'sex': int(request.form.get('sex')),
            'cp': int(request.form.get('cp')),
            'trestbps': float(request.form.get('trestbps')),
            'chol': float(request.form.get('chol')),
            'fbs': int(request.form.get('fbs')),
            'restecg': int(request.form.get('restecg')),
            'thalach': float(request.form.get('thalach')),
            'exang': int(request.form.get('exang')),
            'oldpeak': float(request.form.get('oldpeak')),
            'slope': int(request.form.get('slope')),
            'ca': float(request.form.get('ca')),
            'thal': float(request.form.get('thal'))
        }

        print("\n" + "="*70)
        print("🔍 KULLANICI VERİSİ:")
        print(user_data)
        
        # DataFrame'e çevir
        df = pd.DataFrame([user_data])
        
        # Eksik değerleri doldur
        df_imputed = imputer.transform(df)
        df = pd.DataFrame(df_imputed, columns=features)
        
        # Scale et
        df_scaled = scaler.transform(df)
        
        # Tahmin
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0][1] * 100
        
        print(f"\n🎯 TAHMİN:")
        print(f"  Prediction: {prediction}")
        print(f"  Olasılık: {probability:.2f}%")
        print("="*70 + "\n")
        
        # Risk seviyesi
        risk_level = "high" if prediction == 1 else "low"
        result_title = "Kalp Hastalığı Riski Tespit Edildi" if prediction == 1 else "Kalp Hastalığı Riski Tespit Edilmedi"
        
        # Risk faktörleri
        risk_factors = []
        
        if user_data['age'] > 60:
            risk_factors.append(f"İleri yaş ({user_data['age']})")
        
        if user_data['sex'] == 1:
            risk_factors.append("Erkek cinsiyet (yüksek risk)")
        
        if user_data['cp'] == 0:
            risk_factors.append("Asemptomatik göğüs ağrısı (yüksek risk)")
        
        if user_data['trestbps'] > 140:
            risk_factors.append(f"Yüksek kan basıncı ({user_data['trestbps']} mmHg)")
        
        if user_data['chol'] > 240:
            risk_factors.append(f"Yüksek kolesterol ({user_data['chol']} mg/dl)")
        
        if user_data['fbs'] == 1:
            risk_factors.append("Yüksek açlık kan şekeri")
        
        if user_data['exang'] == 1:
            risk_factors.append("Egzersiz kaynaklı anjina")
        
        if user_data['thalach'] < 100:
            risk_factors.append(f"Düşük maksimum kalp atış hızı ({user_data['thalach']} bpm)")
        
        if user_data['oldpeak'] > 2:
            risk_factors.append(f"Yüksek ST depresyonu ({user_data['oldpeak']})")
        
        if user_data['ca'] > 0:
            risk_factors.append(f"Tıkalı ana damar ({int(user_data['ca'])} adet)")
        
        if user_data['thal'] != 0:
            risk_factors.append("Talasemi anomalisi mevcut")
        
        return render_template(
            "result.html",
            sonuc_baslik=result_title,
            risk_level=risk_level,
            probability=probability,
            risk_factors=risk_factors,
            metrics=metrics
        )
        
    except Exception as e:
        import traceback
        print("\n❌ HATA:")
        print(traceback.format_exc())
        return render_template(
            "result.html",
            sonuc_baslik=f"Hata: {str(e)}",
            risk_level="low",
            probability=None,
            risk_factors=[],
            metrics=None
        )

if __name__ == "__main__":
    app.run(debug=True, port=5001)