import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Memuat model, scaler, dan label encoder yang sudah dilatih sebelumnya di notebook
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')

st.title("Prediksi Status Mahasiswa - Jaya Jaya Institut")
st.write("Aplikasi ini memprediksi apakah seorang mahasiswa berpotensi Dropout, Enrolled, atau Graduate berdasarkan data akademik dan sosio-ekonominya.")

st.header("Masukkan Data Mahasiswa")

marital_status = st.selectbox("Status Pernikahan (1=Single,2=Married,3=Widower,4=Divorced,5=Facto Union,6=Legally Separated)", [1,2,3,4,5,6])
application_mode = st.number_input("Application Mode", min_value=1, max_value=60, value=17)
course = st.number_input("Kode Course", min_value=1, max_value=10000, value=171)
daytime = st.selectbox("Kelas (1=Daytime, 0=Evening)", [1, 0])
prev_qual_grade = st.number_input("Nilai Kualifikasi Sebelumnya", min_value=0.0, max_value=200.0, value=122.0)
admission_grade = st.number_input("Nilai Masuk (Admission Grade)", min_value=0.0, max_value=200.0, value=127.3)
displaced = st.selectbox("Displaced (1=Ya, 0=Tidak)", [1, 0])
special_needs = st.selectbox("Kebutuhan Khusus Pendidikan (1=Ya, 0=Tidak)", [0, 1])
debtor = st.selectbox("Memiliki Tunggakan (1=Ya, 0=Tidak)", [0, 1])
tuition_ok = st.selectbox("UKT Terbayar Tepat Waktu (1=Ya, 0=Tidak)", [1, 0])
gender = st.selectbox("Gender (1=Laki-laki, 0=Perempuan)", [1, 0])
scholarship = st.selectbox("Penerima Beasiswa (1=Ya, 0=Tidak)", [0, 1])
age = st.number_input("Usia saat Mendaftar", min_value=15, max_value=70, value=20)
international = st.selectbox("Mahasiswa Internasional (1=Ya, 0=Tidak)", [0, 1])

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
        marital_status, application_mode, application_order, course, daytime,
        prev_qual, prev_qual_grade, nacionality, mothers_qual, fathers_qual,
        mothers_occ, fathers_occ, admission_grade, displaced, special_needs,
        debtor, tuition_ok, gender, scholarship, age, international,
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
    prediction = model.predict(input_scaled)
    prediction_label = label_encoder.inverse_transform(prediction)[0]
    proba = model.predict_proba(input_scaled)[0]

    st.success(f"Prediksi Status: **{prediction_label}**")
    st.write("Probabilitas tiap kelas:")
    for cls, p in zip(label_encoder.classes_, proba):
        st.write(f"- {cls}: {p:.2%}")