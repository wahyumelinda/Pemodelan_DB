import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
# -------------------- KONFIGURASI --------------------
st.set_page_config(page_title="Rekomendasi Teknisi", page_icon="👨‍🔧", layout="wide")
st.markdown(
        "<div style='background-color: #1e3a8a; padding: 1.5rem; border-radius: 10px; text-align: center;'>"
        "<h1 style='color: white;'>🔧 Rekomendasi Teknisi Perbaikan Mesin 🔧</h1>"
        "<p style='color: white; font-size: 16px;'>Gunakan form di bawah ini untuk memprediksi teknisi yang akan ditugaskan</p>"
        "</div>", unsafe_allow_html=True)

# Load model dan encoder
model = joblib.load("model_teknisi_rf.joblib")
le_target = joblib.load("label_encoder_target.pkl")
df = pd.read_excel("Pemodelan_DB.xlsx")

# Encode fitur
features = ['Nama Mesin', 'No Mesin', 'Masalah', 'Penyebab']
encoded_df = df[features + ['Nama Teknisi', 'Durasi Pengerjaan']].dropna()
label_encoders = {}
for col in features:
    le = LabelEncoder()
    encoded_df[col] = le.fit_transform(encoded_df[col])
    label_encoders[col] = le

st.write("---")

# Input dari user
col1, col2 = st.columns(2)
with col1:
    nama_mesin = st.selectbox("Nama Mesin", df['Nama Mesin'].dropna().unique())
    masalah = st.selectbox("Masalah", df['Masalah'].dropna().unique())
with col2:
    no_mesin = st.selectbox("No Mesin", df['No Mesin'].dropna().unique())
    penyebab = st.selectbox("Penyebab", df['Penyebab'].dropna().unique())

if st.button("🔍 Cari Teknisi Terbaik"):
    try:
        # Transform input
        input_data = {
            'Nama Mesin': [label_encoders['Nama Mesin'].transform([nama_mesin])[0]],
            'No Mesin': [label_encoders['No Mesin'].transform([no_mesin])[0]],
            'Masalah': [label_encoders['Masalah'].transform([masalah])[0]],
            'Penyebab': [label_encoders['Penyebab'].transform([penyebab])[0]],
        }
        input_df = pd.DataFrame(input_data)

        # Prediksi probabilitas dan ambil top 3
        y_proba = model.predict_proba(input_df)
        top3_idx = np.argsort(y_proba, axis=1)[:, -3:][:, ::-1]
        top3_teknisi = [le_target.classes_[i] for i in top3_idx[0]]

        st.write("---")

        st.markdown(
        "<div style='background-color: #4682B4; padding: 1.5rem; border-radius: 10px; text-align: center;'>"
        "<h2 style='color: white;'>👨‍🔧 Rekomendasi Teknisi 👨‍🔧</h2>"
        "</div>", unsafe_allow_html=True)
        for i, teknisi in enumerate(top3_teknisi, 1):
            st.markdown(
                f"""
                <div style='
                    background-color: #f0f4f8;
                    padding: 1rem;
                    border-radius: 10px;
                    margin-bottom: 1rem;
                    box-shadow: 2px 2px 8px rgba(0,0,0,0.05); text-align: center;
                '>
                    <h3 style='margin: 0'>{i}. {teknisi}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("---")
        # Tampilkan data historis
        st.subheader("📁 Data Historis Matching")
        filter_df = df[
            (df['Nama Mesin'] == nama_mesin) &
            (df['No Mesin'] == no_mesin) &
            (df['Masalah'] == masalah) &
            (df['Penyebab'] == penyebab)
        ][['Nama Teknisi', 'Durasi Pengerjaan']]

        if filter_df.empty:
            st.info("Tidak ditemukan histori teknisi untuk kombinasi input tersebut.")
        else:
            min_teknisi = filter_df.sort_values(by='Durasi Pengerjaan').head(1).iloc[0]
            st.dataframe(filter_df.sort_values(by='Durasi Pengerjaan').reset_index(drop=True))
            st.success(f"✅ Teknisi tercepat berdasarkan histori: **{min_teknisi['Nama Teknisi']}** dengan durasi **{min_teknisi['Durasi Pengerjaan']} menit**")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses: {e}")
