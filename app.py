import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Menentukan Promosi Tidaknya Karyawan",
    page_icon=":bar_chart:"
)

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, 'model_kinerja_karyawan')

# Muat model dengan error handling
model = None
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Gagal memuat model dari: {model_path}")
    st.exception(e)
    st.stop()

# Design Streamlit
st.title("🎯 Prediksi Promosi Karyawan Berdasarkan Kinerja dan Faktor Lainnya")
st.markdown("Aplikasi ini memprediksi apakah seorang karyawan layak untuk dipromosikan berdasarkan faktor-faktor seperti usia, lama bekerja, penilaian kinerja, tingkat kepuasan kerja, jumlah pelatihan yang diikuti, dan absensi.")   

st.header("Masukkan Data Karyawan:")
lama_kerja_tahun = st.slider("Lama Bekerja (tahun)", 0, 40, 5)
rata_rata_rating = st.slider("Rata-rata Penilaian Kinerja (1-5)", 1.0, 5.0, 3.0)
jumlah_pelatihan = st.slider("Jumlah Pelatihan yang Diikuti", 0, 40, 5)
departemen = st.selectbox("Departemen", ["Tech", "Sales", "HR"], index=1)
gaji_kenaikan_usd = st.slider("Gaji Kenaikan (USD)", 0, 20000, 5000)

if st.button("Prediksi Promosi Karyawan"):
    try:
        # Siapkan data input dengan urutan kolom SAMA PERSIS dengan training
        # Urutan penting: lama_kerja_tahun, rata_rata_rating, jumlah_pelatihan, gaji_kenaikan_usd, departemen
        data_baru = pd.DataFrame([[lama_kerja_tahun, rata_rata_rating, jumlah_pelatihan, gaji_kenaikan_usd, departemen]], 
                                 columns=['lama_kerja_tahun', 'rata_rata_rating', 'jumlah_pelatihan', 'gaji_kenaikan_usd', 'departemen'])
        
        # Prediksi
        prediksi = model.predict(data_baru)[0]
        
        # Coba dapatkan confidence dari predict_proba jika tersedia
        presentase = None
        try:
            proba_result = model.predict_proba(data_baru)
            if proba_result is not None:
                presentase = max(proba_result[0])
        except (AttributeError, TypeError):
            pass
        
        # Tampilkan hasil
        if prediksi == "Ya":
            if presentase is not None:
                st.success(f"✅ Karyawan diprediksi **LAYAK** untuk dipromosikan dengan tingkat keyakinan **{presentase*100:.2f}%**")
            else:
                st.success(f"✅ Karyawan diprediksi **LAYAK** untuk dipromosikan")
        else:
            if presentase is not None:
                st.warning(f"⚠️ Karyawan diprediksi **TIDAK LAYAK** untuk dipromosikan dengan tingkat keyakinan **{presentase*100:.2f}%**")
            else:
                st.warning(f"⚠️ Karyawan diprediksi **TIDAK LAYAK** untuk dipromosikan")
                
    except Exception as e:
        st.error("❌ Terjadi kesalahan saat melakukan prediksi")
        st.error(f"Pesan error: {str(e)}")
        st.exception(e)

st.divider()
st.caption("Dikembangkan oleh Hana Rohadah @2025")