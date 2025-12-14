# 🫀 Kalp Hastalığı Risk Tahmin Modeli

UCI Heart Disease veri seti ile %86 doğrulukla kalp hastalığı riski tahmin eden yapay zeka modeli.

## 📊 Model Performansı
- **Accuracy:** 86.84%
- **Precision:** 83.78%
- **Recall:** 88.57%
- **F1-Score:** 86.11%
- **ROC-AUC:** 0.9359

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
```bash
pip install -r requirements.txt
```

### Modeli Eğit
```bash
python model.py
```

Bu komut şunları oluşturur:
- `uci_model.pkl`
- `uci_scaler.pkl`
- `uci_imputer.pkl`
- `uci_features.pkl`
- `uci_metrics.pkl`

### Uygulamayı Başlat
```bash
python app.py
```

Tarayıcıda `http://localhost:5001` adresine git.

## 📁 Proje Yapısı
```
heart-disease-prediction/
├── app.py                  # Flask web uygulaması
├── model.py                # Model eğitim scripti
├── uci_heart_disease.csv   # UCI veri seti
├── templates/
│   ├── index.html          # Kullanıcı formu
│   └── result.html         # Tahmin sonucu
├── static/
│   └── style.css           # Stiller
└── requirements.txt        # Python bağımlılıkları
```

## 🛠️ Kullanılan Teknolojiler
- Python 3.13
- Flask (Web framework)
- Scikit-learn (Random Forest)
- Pandas & NumPy
- HTML/CSS

## 📝 Veri Seti
- **Kaynak:** UCI Machine Learning Repository
- **Hasta Sayısı:** 303
- **Özellik Sayısı:** 13
- **Hedef:** Kalp hastalığı var/yok (binary classification)

## 👥 Özellikler
- Yaş, cinsiyet, göğüs ağrısı tipi
- Kan basıncı, kolesterol
- EKG sonuçları
- Maksimum kalp atış hızı
- Egzersiz anjinası
- ST depresyonu, eğim
- Damar sayısı, talasemi

## 📄 Lisans
MIT License

## 👤 Geliştirici
Emre Demir