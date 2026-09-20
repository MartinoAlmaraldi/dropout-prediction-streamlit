import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

st.title("Prediksi Status Mahasiswa - Jaya Jaya Institut")
st.write("Aplikasi ini memprediksi apakah seorang mahasiswa berpotensi Dropout atau Graduate berdasarkan data akademik dan sosio-ekonominya.")

st.header("Masukkan Data Mahasiswa")

# Mapping teks deskriptif ke kode numerik yang dipakai model (dilakukan internal, tidak terlihat pengguna)
marital_map = {"Single": 1, "Married": 2, "Widower": 3, "Divorced": 4, "Facto Union": 5, "Legally Separated": 6}
yes_no_map = {"Ya": 1, "Tidak": 0}
gender_map = {"Laki-laki": 1, "Perempuan": 0}
daytime_map = {"Kelas Pagi/Siang (Daytime)": 1, "Kelas Malam (Evening)": 0}

application_mode_map = {
    "1st phase - general contingent": 1,
    "Ordinance No. 612/93": 2,
    "1st phase - special contingent (Azores Island)": 5,
    "Holders of other higher courses": 7,
    "Ordinance No. 854-B/99": 10,
    "International student (bachelor)": 15,
    "1st phase - special contingent (Madeira Island)": 16,
    "2nd phase - general contingent": 17,
    "3rd phase - general contingent": 18,
    "Ordinance No. 533-A/99, item b2) (Different Plan)": 26,
    "Ordinance No. 533-A/99, item b3 (Other Institution)": 27,
    "Over 23 years old": 39,
    "Transfer": 42,
    "Change of course": 43,
    "Technological specialization diploma holders": 44,
    "Change of institution/course": 51,
    "Short cycle diploma holders": 53,
    "Change of institution/course (International)": 57,
}

course_map = {
    "Biofuel Production Technologies": 33,
    "Animation and Multimedia Design": 171,
    "Social Service (evening attendance)": 8014,
    "Agronomy": 9003,
    "Communication Design": 9070,
    "Veterinary Nursing": 9085,
    "Informatics Engineering": 9119,
    "Equinculture": 9130,
    "Management": 9147,
    "Social Service": 9238,
    "Tourism": 9254,
    "Nursing": 9500,
    "Oral Hygiene": 9556,
    "Advertising and Marketing Management": 9670,
    "Journalism and Communication": 9773,
    "Basic Education": 9853,
    "Management (evening attendance)": 9991,
}

marital_label = st.selectbox("Status Pernikahan", list(marital_map.keys()))
application_mode_label = st.selectbox("Jalur Pendaftaran (Application Mode)", list(application_mode_map.keys()))
course_label = st.selectbox("Program Studi (Course)", list(course_map.keys()))
daytime_label = st.selectbox("Jenis Kelas", list(daytime_map.keys()))
prev_qual_grade = st.number_input("Nilai Kualifikasi Sebelumnya", min_value=0.0, max_value=200.0, value=122.0)
admission_grade = st.number_input("Nilai Masuk (Admission Grade)", min_value=0.0, max_value=200.0, value=127.3)
displaced_label = st.selectbox("Mahasiswa Displaced (pindah tempat tinggal untuk kuliah)", list(yes_no_map.keys()))
special_needs_label = st.selectbox("Kebutuhan Khusus Pendidikan", list(yes_no_map.keys()), index=1)
debtor_label = st.selectbox("Memiliki Tunggakan", list(yes_no_map.keys()), index=1)
tuition_ok_label = st.selectbox("UKT Terbayar Tepat Waktu", list(yes_no_map.keys()))
gender_label = st.selectbox("Gender", list(gender_map.keys()))
scholarship_label = st.selectbox("Penerima Beasiswa", list(yes_no_map.keys()), index=1)
age = st.number_input("Usia saat Mendaftar", min_value=15, max_value=70, value=20)
international_label = st.selectbox("Mahasiswa Internasional", list(yes_no_map.keys()), index=1)

st.subheader("Nilai Akademik Semester 1 & 2")
sem1_credited = st.number_input("SKS Diakui Semester 1", min_value=0, value=0)
sem1_enrolled = st.number_input("SKS Diambil Semester 1", min_value=0, value=6)
sem1_eval = st.number_input("Jumlah Evaluasi Semester 1", min_value=0, value=6)
sem1_approved = st.number_input("SKS Lulus Semester 1", min_value=0, value=6)
sem1_grade = st.number_input("Rata-rata Nilai Semester 1", min_value=0.0, max_value=20.0, value=13.0)
sem1_no_eval = st.number_input("SKS Tanpa Evaluasi Semester 1", min_value=0, value=0)
sem2_credited = st.number_input("SKS Diakui Semester 2", min_value=0, value=0)
sem2_enrolled = st.number_input("SKS Diambil Semester 2", min_value=0, value=6)
sem2_eval = st.number_input("Jumlah Evaluasi Semester 2", min_value=0, value=6)
sem2_approved = st.number_input("SKS Lulus Semester 2", min_value=0, value=6)
sem2_grade = st.number_input("Rata-rata Nilai Semester 2", min_value=0.0, max_value=20.0, value=13.0)
sem2_no_eval = st.number_input("SKS Tanpa Evaluasi Semester 2", min_value=0, value=0)

st.subheader("Indikator Ekonomi")
unemployment = st.number_input("Tingkat Pengangguran (%)", value=10.8)
inflation = st.number_input("Tingkat Inflasi (%)", value=1.4)
gdp = st.number_input("GDP", value=1.74)

nacionality = 1
mothers_qual = 19
fathers_qual = 12
mothers_occ = 5
fathers_occ = 9
prev_qual = 1
application_order = 1

if st.button("Prediksi Status"):
    input_data = pd.DataFrame([[
        marital_map[marital_label], application_mode_map[application_mode_label], application_order,
        course_map[course_label], daytime_map[daytime_label],
        prev_qual, prev_qual_grade, nacionality, mothers_qual, fathers_qual,
        mothers_occ, fathers_occ, admission_grade, yes_no_map[displaced_label], yes_no_map[special_needs_label],
        yes_no_map[debtor_label], yes_no_map[tuition_ok_label], gender_map[gender_label], yes_no_map[scholarship_label],
        age, yes_no_map[international_label],
        sem1_credited, sem1_enrolled, sem1_eval, sem1_approved, sem1_grade, sem1_no_eval,
        sem2_credited, sem2_enrolled, sem2_eval, sem2_approved, sem2_grade, sem2_no_eval,
        unemployment, inflation, gdp
    ]], columns=[
        'Marital_status','Application_mode','Application_order','Course','Daytime_evening_attendance',
        'Previous_qualification','Previous_qualification_grade','Nacionality','Mothers_qualification',
        'Fathers_qualification','Mothers_occupation','Fathers_occupation','Admission_grade','Displaced',
        'Educational_special_needs','Debtor','Tuition_fees_up_to_date','Gender','Scholarship_holder',
        'Age_at_enrollment','International','Curricular_units_1st_sem_credited','Curricular_units_1st_sem_enrolled',
        'Curricular_units_1st_sem_evaluations','Curricular_units_1st_sem_approved','Curricular_units_1st_sem_grade',
        'Curricular_units_1st_sem_without_evaluations','Curricular_units_2nd_sem_credited',
        'Curricular_units_2nd_sem_enrolled','Curricular_units_2nd_sem_evaluations','Curricular_units_2nd_sem_approved',
        'Curricular_units_2nd_sem_grade','Curricular_units_2nd_sem_without_evaluations',
        'Unemployment_rate','Inflation_rate','GDP'
    ])

    input_scaled = scaler.transform(input_data)
    dropout_proba = model.predict_proba(input_scaled)[0][1]
    graduate_proba = 1 - dropout_proba

    if dropout_proba >= 0.7:
        risk_level = "High"
    elif dropout_proba >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    prediction_label = "Dropout" if dropout_proba >= 0.5 else "Graduate"

    st.success(f"Prediksi Status: **{prediction_label}**")
    st.write(f"Graduate Probability: **{graduate_proba:.2%}**")
    st.write(f"Dropout Probability: **{dropout_proba:.2%}**")
    st.write(f"Risk Level: **{risk_level}**")
