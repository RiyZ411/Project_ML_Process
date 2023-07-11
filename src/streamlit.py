import streamlit as st
import requests
import joblib
from PIL import Image

# Load and set images in the first place
header_images = Image.open('assets/monas.jpg')
st.image(header_images)

# Add some information about the service
st.title("Prediksi Polusi Udara Jakarta")
st.write('Project ini dibuat untuk memenuhi project akhir kelas Machine Learning Process Pacmann')
st.write('\t - Nama  : Riyan Zaenal Arifin')
st.write('\t - Email : riyanzaenal411@gmail.com')
st.subheader("Just enter variabel below then click Predict button :sunglasses:")

# Create form of input
with st.form(key = "air_data_form"):
    # Create box for number input
    stasiun = st.number_input(
        label = "1.\tEnter Stasiun Value:",
        min_value = 0,
        max_value = 4,
        help = "Value range from 0 to 4:" 
                "\n- 0 : DKI1 (Bunderan HI)"
                "\n- 1 : DKI2 (Kelapa Gading)"
                "\n- 2 : DKI3 (Jagakarsa)"
                "\n- 3 : DKI4 (Lubang Buaya)"
                "\n- 4 : DKI5 (Kebon Jeruk) Jakarta Barat"
    )

    pm10 = st.number_input(
        label = "2.\tEnter pm10 Value:",
        min_value = 15,
        max_value = 179,
        help = "Value range from 15 to 179"
    )
    
    pm25 = st.number_input(
        label = "3.\tEnter pm25 Value:",
        min_value = 20,
        max_value = 174,
        help = "Value range from 20 to 174"
    )

    so2 = st.number_input(
        label = "4.\tEnter so2 Value:",
        min_value = 4,
        max_value = 82,
        help = "Value range from 4 to 82"
    )

    co = st.number_input(
        label = "5.\tEnter co Value:",
        min_value = 2,
        max_value = 30,
        help = "Value range from 2 to 30"
    )

    o3 = st.number_input(
        label = "6.\tEnter o3 Value:",
        min_value = 8,
        max_value = 81,
        help = "Value range from 8 to 81"
    )

    no2 = st.number_input(
        label = "7.\tEnter no2 Value:",
        min_value = 4,
        max_value = 65,
        help = "Value range from 4 to 65"
    )

    max = st.number_input(
        label = "8.\tEnter max Value:",
        min_value = 26,
        max_value = 179,
        help = "Value range from 26 to 179"
    )

    critical = st.number_input(
        label = "9.\tEnter critical Value:",
        min_value = 0,
        max_value = 3,
        help = "Value range from 0 to 3:"
                "\n- 0 : PM25"
                "\n- 1 : SO2"
                "\n- 2 : PM10"
                "\n- 3 : O3"
    )
    
    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "stasiun": stasiun,
            "pm10": pm10,
            "pm25": pm25,
            "so2": so2,
            "co": co,
            "o3": o3,
            "no2": no2,
            "max": max,
            "critical": critical
        }

        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post(f"http://api_backend:8080/predict", json = raw_data).json()
            
        # Parse the prediction result
        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            if res["res"] != "Tidak ada API":
                st.warning("Ada API")
                if res['prediction'] == 0:
                    st.success("Kondisi udara diprediksi : TIDAK SEHAT")
                else:
                    st.success("Kondisi udara diprediksi : BAIK")
            else:
                st.success("Tidak ada API")