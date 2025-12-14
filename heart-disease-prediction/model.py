import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
import pickle

print("\n" + "="*70)
print("🚀 UCI HEART DISEASE MODEL EĞİTİMİ")
print("="*70)

# Veriyi yükle
df = pd.read_csv('uci_heart_disease.csv')
print(f"\n📊 Veri yüklendi: {len(df)} satır, {len(df.columns)} kolon")

# Hedef
y = df['target']
X = df.drop('target', axis=1)

print(f"\n🎯 Hedef dağılımı:")
print(y.value_counts())
print(f"  - Sağlıklı (0): {sum(y == 0)} kişi ({sum(y==0)/len(y)*100:.1f}%)")
print(f"  - Hasta (1): {sum(y == 1)} kişi ({sum(y==1)/len(y)*100:.1f}%)")

# Eksik değerleri doldur
print(f"\n🔧 Eksik değerler:")
print(X.isnull().sum()[X.isnull().sum() > 0])

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

print(f"✅ Eksik değerler dolduruldu")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\n📂 Veri bölünmesi:")
print(f"  - Eğitim: {len(X_train)} örnek")
print(f"  - Test: {len(X_test)} örnek")

# Standartlaştırma
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model eğitimi
print("\n🤖 Model eğitiliyor (Random Forest)...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train_scaled, y_train)

print("✅ Eğitim tamamlandı!")

# Değerlendirme
print("\n" + "="*70)
print("📊 MODEL PERFORMANSI")
print("="*70)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = model.score(X_test_scaled, y_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n✅ Accuracy: {accuracy*100:.2f}%")
print(f"🎯 Precision: {precision*100:.2f}%")
print(f"🔍 Recall: {recall*100:.2f}%")
print(f"⚖️ F1-Score: {f1*100:.2f}%")
print(f"📈 ROC-AUC: {roc_auc:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Sağlıklı', 'Hasta']))

print("\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  Gerçek Negatif (Doğru Sağlıklı): {cm[0][0]}")
print(f"  Yanlış Pozitif (Hatalı Hasta): {cm[0][1]}")
print(f"  Yanlış Negatif (Hatalı Sağlıklı): {cm[1][0]}")
print(f"  Gerçek Pozitif (Doğru Hasta): {cm[1][1]}")

# Feature importance
print("\n🔝 En Önemli Özellikler:")
feature_names = X.columns
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in importances.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Metrikleri kaydet
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'total_samples': len(y_test)
}

# Kaydet
print("\n💾 Model kaydediliyor...")
pickle.dump(model, open("uci_model.pkl", "wb"))
pickle.dump(scaler, open("uci_scaler.pkl", "wb"))
pickle.dump(imputer, open("uci_imputer.pkl", "wb"))
pickle.dump(list(feature_names), open("uci_features.pkl", "wb"))
pickle.dump(metrics, open("uci_metrics.pkl", "wb"))

print("✅ Başarıyla kaydedildi!")
print("📦 Dosyalar: uci_model.pkl, uci_scaler.pkl, uci_imputer.pkl, uci_features.pkl, uci_metrics.pkl")
print("\n" + "="*70)
print("🎉 EĞİTİM TAMAMLANDI!")
print("="*70 + "\n")