import pandas as pd
pd.set_option('display.max.columns', None)
df = pd.read_csv('ProyekKinerjaKaryawan/Kinerja_Karyawan_Sintetik.csv')
df.info()
print(df['departemen'].value_counts())

correlation_matrix =  df.corr(numeric_only=True)
print(correlation_matrix)
print(df['promosi_ya_tidak'].value_counts())
print("======== Setelah dianalisis awal, Karyawan dengan Lama Kerja, Jumlah Pelatihan, dan Gaji Kenaikan USD yang lebih tinggi cenderung mendapatkan Promosi ========")
#Memetakan Target

Ya = df[df['promosi_ya_tidak']=="Ya"]
Tidak = df[df['promosi_ya_tidak']=="Tidak"]

#Cek Visualisasi SCatter Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(6,5))
plt.scatter(Ya['lama_kerja_tahun'], Ya['jumlah_pelatihan'], alpha=0.3, color='blue', label= "Promosi Ya")
plt.scatter(Tidak['lama_kerja_tahun'], Tidak['jumlah_pelatihan'], alpha=0.3, color='orange', label="Promosi Tidak")
plt.xlabel("Lama Kerja (Tahun)")
plt.ylabel("Jumlah Pelatihan")
plt.title("Lama Kerja vs Jumlah Pelatihan")
plt.legend()
plt.show()

# Membangun Model Machine Learning
print("======== Membangun Model Machine Learning ========")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

x = df[["lama_kerja_tahun", 'rata_rata_rating', 'jumlah_pelatihan', 'gaji_kenaikan_usd', 'departemen']]
y = df['promosi_ya_tidak']

x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

numeric_kolom = ['lama_kerja_tahun', 'rata_rata_rating', 'jumlah_pelatihan', 'gaji_kenaikan_usd']
categorical_kolom = ['departemen']

preposessing = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), numeric_kolom),
        ("oher", OneHotEncoder(), categorical_kolom)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessing", preposessing),
        ("model", RandomForestClassifier())
    ]
)

model.fit(x_train, y_train) #Model Belajar
y_pred = model.predict(x_test) # Soal
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

data_baru = pd.DataFrame([[5, 4.5, 3, 5000, 'Tech']], 
                         columns=['lama_kerja_tahun', 'rata_rata_rating', 'jumlah_pelatihan', 'gaji_kenaikan_usd', 'departemen'])

print(model.predict(data_baru)[0])
print(model.predict_proba(data_baru)[0])

prediksi = model.predict(data_baru)[0]
presentase = max(model.predict_proba(data_baru)[0])
print(f"Prediksi Promosi: {prediksi} dengan tingkat keyakinan {presentase*100:.2f}%")

# Simpan Model
print("======== Menyimpan Model ========")

import joblib
joblib.dump(model, "model_kinerja_karyawan")