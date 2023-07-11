# Project Machine Learning Process PacmannAI - Architectur
- Nama  = Riyan Zaenal Arifin
- Email = riyanzaenal411@gmail.com

## Tahap Persiapan Data
Import library yang dibutuhkan
```python
import pandas as pd
```

Import dataset
```PYTHON
df_juni = pd.read_csv('/home/riyan/MLProject/pencemaran_udara_jakarta/data/raw/indeks-standar-pencemar-udara-di-spku-bulan-juni-tahun-2021.csv')
df_juli = pd.read_csv('/home/riyan/MLProject/pencemaran_udara_jakarta/data/raw/indeks-standar-pencemar-udara-di-spku-bulan-juli-tahun-2021.csv')
df_agust = pd.read_csv('/home/riyan/MLProject/pencemaran_udara_jakarta/data/raw/indeks-standar-pencemar-udara-di-spku-bulan-agustus-tahun-2021.csv')
df_sept = pd.read_csv('/home/riyan/MLProject/pencemaran_udara_jakarta/data/raw/indeks-standar-pencemar-udara-di-spku-bulan-september-tahun-2021.csv')
df_okt = pd.read_csv('/home/riyan/MLProject/pencemaran_udara_jakarta/data/raw/indeks-standar-pencemar-udara-di-spku-bulan-oktober-tahun-2021.csv')
df_nov = pd.read_csv('/home/riyan/MLProject/pencemaran_udara_jakarta/data/raw/indeks-standar-pencemar-udara-di-spku-bulan-november-tahun-2021.csv')
df_desmb = pd.read_csv('/home/riyan/MLProject/pencemaran_udara_jakarta/data/raw/indeks-standar-pencemar-udara-di-spku-bulan-desember-tahun-2021.csv')
```

Rename column name juni and juli
```python
df_juni.columns=df_desmb.columns
df_juli.columns=df_desmb.columns
```

Gabungkan dataset
```python
df = pd.concat([df_juni, df_juli, df_agust, df_sept, df_okt, df_nov, df_desmb])
```

Rename value SEDANG to BAIK in column cetagori
```python
df['categori'] = df['categori'].replace(['SEDANG'], 'BAIK')
```

Simpan ke folder raw dengan format csv
```python
df.to_csv('/home/riyan/MLProject/pencemaran_udara_jakarta/data/raw/rawdataset', index=False)
df
```
![preparation](https://github.com/RiyZ411/Project_ML_Process/blob/main/Gambar/Preparation/1.png)

## Tahap Preprocessing Data
### Import library
```python
import pandas as pd
import numpy as np
import src.util as utils
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
```
### Load Configuration File
```PYTHON
config = utils.load_config()
```
### Load Dataset
Masukan nama variable yang menyimpan path csv di file yaml.
```python
dataset = pd.read_csv(config["dataset_path"])
dataset
```
### Data Clansing
Ada beberapa tahapan dalam pembersihan data, seperti:
- Remove missing value:
    ```python
    #definisikan nilai missing yang kemungkinan terjadi
    missing_values = ['', ' ', 'NaN', 'Nan', 'nan', '.', ',','---']
    col_names = list(dataset.columns)
    dataset[col_names] = dataset[col_names].replace(missing_values, np.nan)
    #cek jumlah missing
    dataset.isna().sum()
    ```
    ![rmissing]()
    Hapus missing value
    ```python
    dataset = dataset.dropna()
    dataset.isna().sum()
    ```
    ![rmissing]()
- Feature Engineering - Feature Selection:

    Karena tanggal tidak dimasukan di model klasifikasi, maka dihapus:
    ```PYTHON
    dataset = dataset.drop(['tanggal'], axis=1)
    dataset
    ```
    ![feature]()
- Label Encoding:

    Agar lebih optimal dalam balancing data menggunakan smote dan pada saat pemodelan, data kategori lebih baik diubah menjadi angka.
    ```PYTHON
    label_encoder = preprocessing.LabelEncoder()

    #kolom stasiun
    dataset['stasiun']= label_encoder.fit_transform(dataset['stasiun'])
    
    #kolom critical
    dataset['critical']= label_encoder.fit_transform(dataset['critical'])
    ```
    Khusus untuk kolom critical ubah BAIK menjadi 1 dan TIDAK SEHAT MENJADI 0.

    ```PYTHON
    #kolom categori
    dataset['categori'] = dataset['categori'].replace(['BAIK', 'TIDAK SEHAT'],[1, 0])
    dataset
    ```
    ![label]()


- Balancing data

    Perlu diketahui juga bahwa untuk balancing data bisa menggunakan 3 metode, yaitu udersampling(kategori yang paling banyak akan disamakan dengan cara menurunkan jumlah datanya sebanyak kategori yang paling sedikit), oversampling(kategori yang paling sedikit akan disamakan dengan cara menambahkan jumlah datanya sebanyak kategori yang paling banyak) dan SMOTE (mirip seperti oversapling, namun dalam penambahan datanya mirip cara kerja KNN, yaitu akan mengambil sample acak, lalu akan dipilih tetangga terdekat). Teknik balancing data seperti oversampling dan SMOTE tidak serta merta membuat data langsung balace dengan bertambahnya data, kemungkinan terdapat data yang duplikat, sehingga perlu dicek duplikasi datanya. Teknik balancing ini umumnya untuk membantu mengurangi ketidakseimbangan data yang cukup signifikan.

    ```PYHON
    #smote
    sm = SMOTE(random_state = 42)
    X_res, y_res = sm.fit_resample(dataset.iloc[:,:-1], dataset.iloc[:,-1:])
    dataset = pd.concat([X_res, y_res], axis=1)
    dataset
    ```
    ![smote]()

- Drop duplicate
    ```PYTHON
    dataset.duplicated().sum()
    ```
    ![dup]()

### Data Defense
Buat fungsi untuk mengecek type data yang masuk

- Ubah type data untuk mengetahui nilai min dan max di describe untuk setting range
    ```python
    dataset = dataset.astype(int)
    dataset.dtypes
    ```
- Check describe

    Masukan nilai min dan max ke file yaml dari output python ini. Ini untuk memastikan data yang konsisten:
    ```python
    dataset.describe()
    ```
    ![descrb]()
    
- Buat fungsi check_data
    ```python
    def check_data(input_data, config):
        # Measure the range of input data
        len_input_data = len(input_data)

        # Check data types
        assert input_data.select_dtypes("int").columns.to_list() == config["int_columns"], "an error occurs in int column(s)."

        # Check range of data
        assert input_data[config["int_columns"][0]].between(config["range_stasiun"][0], config["range_stasiun"][1]).sum() == len_input_data, "an error occurs in stasiun range."
        assert input_data[config["int_columns"][1]].between(config["range_pm10"][0], config["range_pm10"][1]).sum() == len_input_data, "an error occurs in pm10 range."
        assert input_data[config["int_columns"][2]].between(config["range_pm25"][0], config["range_pm25"][1]).sum() == len_input_data, "an error occurs in pm25 range."
        assert input_data[config["int_columns"][3]].between(config["range_so2"][0], config["range_so2"][1]).sum() == len_input_data, "an error occurs in so2 range."
        assert input_data[config["int_columns"][4]].between(config["range_co"][0], config["range_co"][1]).sum() == len_input_data, "an error occurs in co range."
        assert input_data[config["int_columns"][5]].between(config["range_o3"][0], config["range_o3"][1]).sum() == len_input_data, "an error occurs in o3 range."
        assert input_data[config["int_columns"][6]].between(config["range_no2"][0], config["range_no2"][1]).sum() == len_input_data, "an error occurs in no2 range."
        assert input_data[config["int_columns"][7]].between(config["range_max"][0], config["range_max"][1]).sum() == len_input_data, "an error occurs in max range."
        assert input_data[config["int_columns"][8]].between(config["range_critical"][0], config["range_critical"][1]).sum() == len_input_data, "an error occurs in critical range."
        assert input_data[config["int_columns"][9]].between(config["range_categori"][0], config["range_categori"][1]).sum() == len_input_data, "an error occurs in categori range."

- Jalankan fungsi menggunakan dataset:
    ```python
    check_data(dataset, config)
    ```


### Data Splitting
- Load kolom label dan predictors yang sudah didefinisikan di file yaml.
    ```python
    x = dataset[config["predictors"]].copy()
    y = dataset[config["label"]].copy()
    ```
- Spliting data training, validation, dan testing

    Untuk data testing diambil 10% dari data training
    ```python
    #split data training dan testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42, stratify = y)
    ```
    Untuk data validation diambil 30% dari data training yang sudah dibagi dengan data testing
    ```python
    #split data training dan validation
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.3, random_state = 42, stratify = y_train)
    ```

- Hasil pembagian data
    ```python
    data_split = {
        'Spliting_Data': ['Training', 'Validation', 'Testing'],
        'Values': [len(x_train),len(x_valid),len(x_test)],
    }

    # Create a DataFrame from the dictionary
    data_split = pd.DataFrame(data_split)

    #set frame
    plt.figure(figsize=(10, 6)) 
    ax = sns.barplot(x='Spliting_Data', y='Values', data=data_split)
    plt.bar_label(ax.containers[0])

    # Show the plot
    plt.show()
    ```
    ![split]()

### Save to file yaml.
Simpan pembagian data ke file yaml
```python
utils.pickle_dump(dataset, config["dataset_cleaned_path"])

utils.pickle_dump(x_train, config["train_set_path"][0])
utils.pickle_dump(y_train, config["train_set_path"][1])

utils.pickle_dump(x_valid, config["valid_set_path"][0])
utils.pickle_dump(y_valid, config["valid_set_path"][1])

utils.pickle_dump(x_test, config["test_set_path"][0])
utils.pickle_dump(y_test, config["test_set_path"][1])

```
## Modeling dan Evaluasi
### Persiapan library dan dataset
- Import library yang dibutuhkan:
    ```python
    import src.util as utils
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    ```
- Panggil fungsi load_config di file util:
    ```python 
    config = utils.load_config()
    ```
- Buat fungsi untuk load dataset
    ```python
    def load_train(params: dict) -> pd.DataFrame:
        # Load train set
        x_train = utils.pickle_load(params["train_set_path"][0])
        y_train = utils.pickle_load(params["train_set_path"][1])

        return x_train, y_train

    def load_valid(params: dict) -> pd.DataFrame:
        # Load valid set
        x_valid = utils.pickle_load(params["valid_set_path"][0])
        y_valid = utils.pickle_load(params["valid_set_path"][1])

        return x_valid, y_valid

    def load_test(params: dict) -> pd.DataFrame:
        # Load tets set
        x_test = utils.pickle_load(params["test_set_path"][0])
        y_test = utils.pickle_load(params["test_set_path"][1])

        return x_test, y_test
    ```
- Ambil dataset di file yaml menggunakan fungsi di atas:
    ```python
    x_train, y_train = load_train(config)
    x_valid, y_valid = load_valid(config)
    x_test, y_test = load_test(config)
    ```
### Modeling
- Train Model
    ```python
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    ```
- Plot Tree      
    ![pltree]()

### Evaluasi
- Data Validation
    - Classification Report
        ```python
        y_pred = dtc.predict(x_valid)
        print(classification_report(y_valid, y_pred))
        ```
        ![class]()
    - Confussion Matrix
        ```python
        ConfusionMatrixDisplay.from_predictions(y_valid, y_pred)
        ```
        ![conf]()
    - Metrik
        ```python
        y_pred_val = dtc.predict(x_valid)

        # Calculating accuracy
        accuracy = accuracy_score(y_valid, y_pred_val)
        print("Accuracy:", accuracy)

        # Calculating precision
        precision = precision_score(y_valid, y_pred_val)
        print("Precision:", precision)

        # Calculating recall
        recall = recall_score(y_valid, y_pred_val)
        print("Recall:", recall)

        # Calculating F1 score
        f1 = f1_score(y_valid, y_pred_val)
        print("F1 score:", f1)

        eval_val = {
            'Evaluation Data Validation': ['Accuracy', 'Precision', 'Recall','F1'],
            'Values': [accuracy,precision,recall,f1],
        }

        # Create a DataFrame from the dictionary
        eval_val = pd.DataFrame(eval_val)

        #set frame
        plt.figure(figsize=(10, 6)) 
        ax = sns.barplot(x='Evaluation Data Validation', y='Values', data=eval_val)
        plt.bar_label(ax.containers[0])
        ```
        ![metrix]()
- Data Training
    - Metrik 
        ```python
        # Calculating accuracy
        y_pred_train = dtc.predict(x_train)

        #
        accuracy = accuracy_score(y_train, y_pred_train)
        print("Accuracy:", accuracy)

        # Calculating precision
        precision = precision_score(y_train, y_pred_train)
        print("Precision:", precision)

        # Calculating recall
        recall = recall_score(y_train, y_pred_train)
        print("Recall:", recall)

        # Calculating F1 score
        f1 = f1_score(y_train, y_pred_train)
        print("F1 score:", f1)

        eval_train = {
            'Evaluation Data Training': ['Accuracy', 'Precision', 'Recall','F1'],
            'Values': [accuracy,precision,recall,f1],
        }

        # Create a DataFrame from the dictionary
        eval_train = pd.DataFrame(eval_train)

        #set frame
        plt.figure(figsize=(10, 6)) 
        ax = sns.barplot(x='Evaluation Data Training', y='Values', data=eval_train)
        plt.bar_label(ax.containers[0])

        # Show the plot
        plt.show()
        ```
        ![metrixtrn]()
### Predict Data Testing
Prediksi y_test dari x_test dengan model
```python
#predict data testing dengan membuat kolom baru predict_categori
x_test["predict_categori"] = dtc.predict(x_test)

#gabung untuk membandingkan hasil predicksi dengan y_test 
test = pd.concat([x_test, y_test], axis = 1)
test
```
![predtes]()

### Simpan Model
Simpan model yang sudah fix ke file yaml dengan format pickle (pkl)
```python
utils.pickle_dump(dtc, config["production_model_path"])
model = utils.pickle_load("models/production_model.pkl")
```

## Format message untuk melakukan prediksi via API
### Import library yang dibutuhkan
```PYTHON
import streamlit as st
import requests
import joblib
from PIL import Image
```
Buat form input menggunakan streamlit
```python
# Load and set images in the first place
header_images = Image.open('/home/riyan/MLProject/pencemaran_udara_jakarta/assets/monas.jpg')
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
```
Send input data ke API untuk dilakukan prediksi
```python
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
            res = requests.post(f"http://127.0.0.1:8000/predict", json = raw_data).json()
```
Pastkan tidak ada error menggunakan branching, jika tidak ada pastikan juga ada API atau tidak menggunakan branching. Dikarenakan hasil prediksinya 0 atau 1, untuk mempermudah pemahaman user, hasl dari response API bisa diubah menjadi BAIK untuk 1 dan TIDAK SEHAT UNTUK 0.
```PYTHON            
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
```
## Format message response dari API
### Import library yang dibutuhkan 
```PYTHON
import util as utils
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np
import data_pipeline as data_pipeline
import preprocessing as preprocessing
```
### Load Model
```PYTHON
config = utils.load_config()
model_data = utils.pickle_load(config["production_model_path"])
```

### Definisikan tipe data yang diterima fastAPI
```PYTHON
class api_data(BaseModel):
    stasiun : int
    pm10 : int
    pm25 : int
    so2 : int
    co : int
    o3 : int
    no2 : int
    max : int
    critical : int
```
### Response fastAPI
```PYTHON
app = FastAPI()
```
- Untuk mempermudah mengetahui server fastAPI sudah terhubung, buat "Hello, FastAPI up!":
    ```PYTHON
    @app.get("/")
    def home():
        return "Hello, FastAPI up!"
    ```
- Data yang diterima dari input user akan diprediksi menggunakan model. Sehingga fungsi predict akan mengembalikan hasil prediksi, lalu akan dikirim ke klien server
    ```python
    @app.post("/predict/")
    def predict(data: api_data):    
        # Convert data api to dataframe
        data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)  # type: ignore
        data.columns = config["predictors"]

        # Convert dtype
        data = data.astype(int)

        # Check range data
        try:
            data_pipeline.check_data(data, config, True)  # type: ignore
        except AssertionError as ae:
            return {"res": [], "error_msg": str(ae)}

        # Predict data
        y_pred = model_data.predict(data)
        label = [0,1]
        predict = model_data.predict(data)

        if y_pred[0] is None:
            y_pred = "Tidak ada API"
        else:
            y_pred = "Ada API"
        return {"res" : y_pred, "error_msg": "", "prediction" : label[predict[0]]}

    if __name__ == "__main__":
        uvicorn.run("api:app", host = "0.0.0.0", port = 8000)

    ```
## Cara menjalankan layanan Machine Learning di komputer lokal
### Retraining Model 
Arahkan command ubuntu ke folder python FastAPI, lalu jalankan server FastAPI menggunakan perintah "uvicorn api:app --reload" lalu masuk ke link untuk memastikan sudah terhubung dan akan menampilkan seperti ini:
![fastapi]()

### Running API
Pastikan server fastAPI masih hidup, lalu tambah terminal lalu arahkan command ubuntu ke folder python streamlit, lalu jalankan server streamlit menggunakan perintah "streamlit run streamlit.py" lalu masuk ke link akan menampilkan seperti ini dan sudah bisa digunakan untuk memprediksi kondisi udara:                 
![streamlit]()


## Cara menjalankan layanan Machine Learning via Docker
Docker akan membuat dan menjalankan container untuk masing-masing server tersebut dan bisa dijalankan secara bersama-sama, sehingga tidak perlu menjalankan secara manual satu-satu. Pastikan sudah menginstall docker. Buat dockerfile untuk masing-masing server lalu buat file yaml untuk menjalankan kedua server tersebut.
- API
    ```python
    FROM python:3.9.15-slim-buster
    WORKDIR /home
    COPY ./requirements.txt ./
    RUN \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    pip install --upgrade pip && \
    pip install wheel && \
    pip install -r requirements.txt
    EXPOSE 8080
    CMD ["python", "src/api.py"]
    ```
- Streamlit
    ```python
    FROM python:3.9.15-slim-buster
    WORKDIR /home
    COPY ./requirements.txt ./
    RUN \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    pip install --upgrade pip && \
    pip install wheel && \
    pip install -r requirements.txt
    EXPOSE 8501
    CMD ["streamlit", "run", "src/streamlit.py"]
    ```
Selain itu juga buat file requirements.txt untuk kebutuhan library masing-masing server dalam satu folder masing-masing dockerfile. Selanjutnya buat file yaml untuk membuat container masing-masing server tersebut:
```yaml
name: jakarta_air_quality_prediction
services:
  streamlit:
    build: docker/streamlit
    image: ipincogan/project_ml_process_streamlit
    container_name: streamlit_frontend
    depends_on:
      - api
    ports:
      - 8501:8501
    volumes:
      - ./:/home/
  api:
    build: docker/api
    image: ipincogan/project_ml_process_api
    container_name: api_backend
    ports:
      - 8080:8080
    volumes:
      - ./:/home/

```
Untuk menjalankan docker, kita bisa menjalankannya via lokal komputer atau di cloud provider, seperti menggunakan Virtual Machine dari AWS, yaitu EC2:
### Docker lokal komputer
Untuk menjalankan docker di lokal computer sebagai berikut:
- Masuk ke folder ubuntu docker yang sudah dibuat container
- Aktifkan docker yang sudah diinstall dengan perintah "sudo service docker start"
- Jika sudah aktif, ketik "sudo docker compose build" lalu ketik enter
- Aktifkan docker dengan perintah "sudo docker compose up"             
    ![compose]()
- Lalu masuk ke browser, masukkan link "http://localhost:8501/" jika berhasil, maka akan muncul seperti berikut:         
    ![dockerlokal]()

### Docker AWS EC2
Untuk menjalankan docker di AWS EC2, pastikan sudah mendaftar dan membuat instance EC2 di AWS, sesuaikan OS yang digunakan. Jika sudah, nanti akan terdownload otomatis file dengan format .pem. Simpan baik-baik file tersebut, file tersebut nantinya digunakan untuk mengaktifkan EC2, berikut cara mengaktifkan EC2:
- Pastikan project sudah dipush ke github, sehingga projectnya bisa dicloning ke EC2.
- Untuk mengaktifkan EC2, arahkan command ubuntu ke folder yang menyimpan .pem.

    Jalankan commnd berikut agar key tidak bisa diakses secara publik, sesuaikan nama file .pme:
    ```ubuntu
    chmod 400 pacmann-key-aws.pem 
    ```
    ```ubuntu
    sudo ssh -i "pacmann-key-aws.pem" ubuntu@ec2-54-179-75-24.ap-southeast-1.compute.amazonaws.com
    ```
- Clone project dengan perintah git clone "nama repository"

