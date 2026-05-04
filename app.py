import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback

st.set_page_config(page_title="Prediksi Harga Mobil | The Outlier", layout="centered")

st.title("🚗 Prediksi Harga Mobil Bekas")
st.markdown("**Sedang memeriksa sistem server...**")

# Memaksa server menampilkan error di layar, bukan layar putih
try:
    st.info("Mencoba membaca file model (.pkl)...")
    
    @st.cache_resource
    def load_files():
        model = joblib.load('model_outlier_hgb.pkl')
        kolom = joblib.load('kolom_outlier.pkl')
        encoding = joblib.load('encoding_outlier.pkl')
        med_hp = joblib.load('median_hp_outlier.pkl')
        med_liter = joblib.load('median_liter_outlier.pkl')
        return model, kolom, encoding, med_hp, med_liter

    model, model_columns, te_map, median_hp, median_liter = load_files()
    
    st.success("✅ Semua file model BERHASIL dibaca! Mesin siap digunakan.")
    
    # --- UI SIDEBAR ---
    st.sidebar.header("🔧 Spesifikasi Kendaraan")
    brand = st.sidebar.selectbox("Merek", ["Toyota", "Ford", "Chevrolet", "BMW", "Mercedes-Benz", "Honda", "Nissan", "Lainnya"])
    car_model = st.sidebar.text_input("Nama Model (Cth: F-150, Civic, Mustang)", "Civic")
    year = st.sidebar.slider("Tahun Rilis", 1990, 2024, 2018)
    milage = st.sidebar.number_input("Jarak Tempuh (Mil)", min_value=0, value=50000, step=1000)
    fuel = st.sidebar.selectbox("Tipe Bahan Bakar", ["Gasoline", "Hybrid", "E85 Flex Fuel", "Diesel", "Lainnya"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Riwayat & Mesin**")
    accident = st.sidebar.radio("Riwayat Kecelakaan?", ["Tidak Ada / Tidak Dilaporkan", "Pernah Kecelakaan"])
    clean_title = st.sidebar.radio("Surat Kendaraan Bersih (Clean Title)?", ["Ya", "Tidak"])

    hp_input = st.sidebar.number_input("Horsepower (Kosongkan/0 jika tidak tahu)", min_value=0.0, value=0.0)
    liter_input = st.sidebar.number_input("Kapasitas Mesin/Liter (Kosongkan/0 jika tidak tahu)", min_value=0.0, value=0.0)

    # --- LOGIKA PREDIKSI ---
    if st.button("HITUNG ESTIMASI HARGA", use_container_width=True):
        hp_final = hp_input if hp_input > 0 else median_hp
        liter_final = liter_input if liter_input > 0 else median_liter
        acc_bin = 0 if accident == "Tidak Ada / Tidak Dilaporkan" else 1
        title_bin = 1 if clean_title == "Ya" else 0
        
        car_age = 2024 - year
        car_age = car_age if car_age > 0 else 1 
        milage_per_year = milage / car_age
        
        data_input = {
            'model_year': [year], 'milage': [milage], 'accident_binary': [acc_bin],
            'clean_title_binary': [title_bin], 'brand': [brand], 'model': [car_model],
            'fuel_type': [fuel], 'horsepower': [hp_final], 'engine_liter': [liter_final],
            'car_age': [car_age], 'milage_per_year': [milage_per_year]
        }
        df_input = pd.DataFrame(data_input)
        
        model_name = df_input['model'].iloc[0]
        df_input['model_encoded'] = te_map.get(model_name, te_map['GLOBAL_MEAN'])
        df_input = df_input.drop(columns=['model'])
        
        df_input_ohe = pd.get_dummies(df_input, columns=['brand', 'fuel_type'])
        df_final = df_input_ohe.reindex(columns=model_columns, fill_value=0)
        
        log_prediksi = model.predict(df_final)
        harga_asli = np.expm1(log_prediksi)[0]
        
        st.markdown(f"<h1 style='text-align: center; color: #1E88E5;'>${harga_asli:,.2f}</h1>", unsafe_allow_html=True)

except Exception as e:
    st.error("🚨 GAGAL MEMBACA FILE MODEL!")
    st.write("Tolong salin tulisan merah di bawah ini dan kirimkan ke saya:")
    st.code(traceback.format_exc())
    st.write("Daftar file yang ada di server saat ini:")
    st.write(os.listdir('.'))
