import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. SETTING HALAMAN & HEADER
# ==========================================
st.set_page_config(page_title="Prediksi Harga Mobil | The Outlier", layout="centered")
st.title("🚗 Prediksi Harga Mobil Bekas")
st.markdown("**Final Project Data Science Batch 58 - Group 7: The Outlier**")
st.markdown("---")

# ==========================================
# 2. LOAD MODEL & FILE PENDUKUNG
# ==========================================
@st.cache_resource
def load_files():
    model = joblib.load('model_outlier_hgb.pkl')
    kolom = joblib.load('kolom_outlier.pkl')
    encoding = joblib.load('encoding_outlier.pkl')
    med_hp = joblib.load('median_hp_outlier.pkl')
    med_liter = joblib.load('median_liter_outlier.pkl')
    return model, kolom, encoding, med_hp, med_liter

model, model_columns, te_map, median_hp, median_liter = load_files()

# 3. SIDEBAR (INPUT PENGGUNA)
# ==========================================
st.sidebar.header("🔧 Spesifikasi Kendaraan")

# Kamus Pintar Merek & Model
brand_model_dict = {
    "Toyota": ["Camry", "Corolla", "RAV4", "Highlander", "Tacoma", "Tundra", "Sienna", "Prius", "Yaris", "4Runner"],
    "Ford": ["F-150", "Mustang", "Explorer", "Escape", "Focus", "Fusion", "Ranger", "Edge", "Expedition"],
    "Chevrolet": ["Silverado 1500", "Equinox", "Malibu", "Tahoe", "Cruze", "Camaro", "Colorado", "Traverse", "Impala"],
    "BMW": ["3 Series", "5 Series", "X3", "X5", "M3", "M4", "4 Series", "7 Series", "X1"],
    "Mercedes-Benz": ["C-Class", "E-Class", "S-Class", "GLC", "GLE", "A-Class", "GLA"],
    "Honda": ["Civic", "Accord", "CR-V", "Pilot", "Odyssey", "HR-V", "Fit", "Ridgeline"],
    "Nissan": ["Altima", "Sentra", "Rogue", "Maxima", "Pathfinder", "Murano", "Frontier", "Titan"],
    "Lainnya": ["Lainnya"]
}

# Kamus Rentang Tahun Produksi: "Nama Model" -> (Tahun_Awal, Tahun_Akhir)
# (Mobil yang tidak ditulis di sini otomatis akan diberi rentang standar 1990-2024)
model_year_dict = {
    "Yaris": (2007, 2020), 
    "M4": (2015, 2024), 
    "4 Series": (2014, 2024),
    "GLC": (2016, 2024), 
    "GLE": (2016, 2024), 
    "A-Class": (2019, 2022), 
    "GLA": (2015, 2024),
    "HR-V": (2016, 2024), 
    "Cruze": (2011, 2019), 
    "Focus": (2000, 2018), 
    "Fusion": (2006, 2020)
}

brand = st.sidebar.selectbox("Merek", list(brand_model_dict.keys()))

# Logika Dinamis Dropdown Model & Slider Tahun
if brand == "Lainnya":
    car_model = st.sidebar.text_input("Ketik Nama Merek & Model (Cth: Kia Stinger GT):", "Kia Stinger GT")
    min_year, max_year = 1990, 2024
else:
    car_model = st.sidebar.selectbox("Nama Model", brand_model_dict[brand])
    # Ambil batas tahun dari model_year_dict, kalau tidak ada pakai 1990-2024
    min_year, max_year = model_year_dict.get(car_model, (1990, 2024))

# Slider otomatis mengunci rentang tahun, dan posisi awal di tahun terbarunya
year = st.sidebar.slider("Tahun Rilis", min_value=min_year, max_value=max_year, value=max_year)

milage = st.sidebar.number_input("Jarak Tempuh (Mil)", min_value=0, value=50000, step=1000)
fuel = st.sidebar.selectbox("Tipe Bahan Bakar", ["Gasoline", "Hybrid", "E85 Flex Fuel", "Diesel", "Lainnya"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Riwayat & Mesin**")
accident = st.sidebar.radio("Riwayat Kecelakaan?", ["Tidak Ada / Tidak Dilaporkan", "Pernah Kecelakaan"])
clean_title = st.sidebar.radio("Surat Kendaraan Bersih (Clean Title)?", ["Ya", "Tidak"])

hp_input = st.sidebar.number_input("Horsepower (Kosongkan/0 jika tidak tahu)", min_value=0.0, value=0.0)
liter_input = st.sidebar.number_input("Kapasitas Mesin/Liter (Kosongkan/0 jika tidak tahu)", min_value=0.0, value=0.0)

# ==========================================
# 4. LOGIKA PREDIKSI (TOMBOL KLIK)
# ==========================================
if st.button("HITUNG ESTIMASI HARGA", use_container_width=True):
    
    # Feature Engineering (Otomatis)
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
    
    # Target Encoding
    model_name = df_input['model'].iloc[0]
    df_input['model_encoded'] = te_map.get(model_name, te_map['GLOBAL_MEAN'])
    df_input = df_input.drop(columns=['model'])
    
    # One-Hot Encoding
    df_input_ohe = pd.get_dummies(df_input, columns=['brand', 'fuel_type'])
    df_final = df_input_ohe.reindex(columns=model_columns, fill_value=0)
    
    # Eksekusi Model
    log_prediksi = model.predict(df_final)
    harga_asli = np.expm1(log_prediksi)[0]
    
    # Tampilkan Hasil
    st.success("Analisis selesai! Berdasarkan spesifikasi yang diberikan:")
    st.markdown(f"<h1 style='text-align: center; color: #1E88E5;'>${harga_asli:,.2f}</h1>", unsafe_allow_html=True)
    st.info(f"💡 **Insight Mesin:** Mobil berumur {car_age} tahun ini rata-rata menempuh jarak {milage_per_year:,.0f} mil per tahun.")
