import pandas as pd
import util as utils
import copy
from sklearn.model_selection import train_test_split

def read_raw_data(config: dict) -> pd.DataFrame:
    # Return raw dataset
    return pd.read_csv(config["dataset_path"])

# def convert_datetime(input_data: pd.DataFrame, config: dict) -> pd.DataFrame:
#     input_data = input_data.copy()

#     # Convert to datetime
#     input_data[config["datetime_columns"][0]] = pd.to_datetime(
#             input_data[config["datetime_columns"][0]],
#             unit = "s"
#     )

#     return input_data

def check_data(input_data: pd.DataFrame, config: dict, api: bool = False):
    input_data = copy.deepcopy(input_data)
    config = copy.deepcopy(config)

    if not api:
        # Check column data types
     #    assert input_data.select_dtypes("datetime").columns.to_list() == \
     #        config["datetime_columns"], "an error occurs in datetime column(s)."
        assert input_data.select_dtypes("int").columns.to_list() == \
            config["int_columns"], "an error occurs in int column(s)."


        # Check range of categori
        assert input_data[config["int_columns"][9]].between(
             config["range_categori"][0], 
             config["range_categori"][1]
             ).sum() == len(input_data), "an error occurs in categori range."

    else:
        # In case checking data from api
        # Last 2 column names in list of int columns are not used as predictor (CNT and Fire Alarm)
        int_columns = config["int_columns"]
        del int_columns[-1:]

        # Last 4 column names in list of int columns are not used as predictor (NC2.5, NC1.0, NC0.5, and PM2.5)
        # float_columns = config["float_columns"]
        # del float_columns[-4:]

        # Check column data types
        assert input_data.select_dtypes("int64").columns.to_list() == \
            int_columns, "an error occurs in int column(s)."
        # assert input_data.select_dtypes("float64").columns.to_list() == \
        #     float_columns, "an error occurs in float column(s)."
        
        assert input_data[config["int_columns"][0]].between(
             config["range_stasiun"][0], 
             config["range_stasiun"][1]
             ).sum() == len(input_data), "an error occurs in stasiun range."
        
        assert input_data[config["int_columns"][1]].between(
             config["range_pm10"][0], 
             config["range_pm10"][1]
             ).sum() == len(input_data), "an error occurs in pm10 range."
        
        assert input_data[config["int_columns"][2]].between(
             config["range_pm25"][0], 
             config["range_pm25"][1]
             ).sum() == len(input_data), "an error occurs in pm25 range."
        
        assert input_data[config["int_columns"][3]].between(
             config["range_so2"][0], 
             config["range_so2"][1]
             ).sum() == len(input_data), "an error occurs in so2 range."
        
        assert input_data[config["int_columns"][4]].between(
             config["range_co"][0], 
             config["range_co"][1]
             ).sum() == len(input_data), "an error occurs in co range."
        
        assert input_data[config["int_columns"][5]].between(
             config["range_o3"][0], 
             config["range_o3"][1]
             ).sum() == len(input_data), "an error occurs in o3 range."
        
        assert input_data[config["int_columns"][6]].between(
             config["range_no2"][0], 
             config["range_no2"][1]
             ).sum() == len(input_data), "an error occurs in no2 range."
        
        assert input_data[config["int_columns"][7]].between(
             config["range_max"][0], 
             config["range_max"][1]
             ).sum() == len(input_data), "an error occurs in max range."
        
        assert input_data[config["int_columns"][8]].between(
             config["range_critical"][0], 
             config["range_critical"][1]
             ).sum() == len(input_data), "an error occurs in critical range."
    
       

def split_data(input_data: pd.DataFrame, config: dict):
# Split predictor and label
        x = input_data[config["predictors"]].copy()
        y = input_data[config["label"]].copy()

        # 1st split train and test
        x_train, x_test, \
        y_train, y_test = train_test_split(
                x, y,
                test_size = config["test_size"],
                random_state = 42,
                stratify = y
        )

        # 2nd split test and valid
        x_train, x_valid, \
        y_train, y_valid = train_test_split(
                x_train, y_train,
                test_size = config["valid_size"],
                random_state = 42,
                stratify = y_valid
        )

        return x_train, x_valid, x_test, y_train, y_valid, y_test


if __name__ == "__main__":
    # 1. Load configuration file
    config = utils.load_config()

    # 2. Read all raw dataset
    raw_dataset = read_raw_data(config)

    # 3. Convert to datetime
    #raw_dataset = convert_datetime(raw_dataset, config)

    # 4. Data defense for non API data
    check_data(raw_dataset, config)

    # 5. Splitting train, valid, and test set
    x_train, x_valid, x_test, \
        y_train, y_valid, y_test = split_data(raw_dataset, config)

    # 6. Save train, valid and test set
    utils.pickle_dump(x_train, config["train_set_path"][0])
    utils.pickle_dump(y_train, config["train_set_path"][1])

    utils.pickle_dump(x_valid, config["valid_set_path"][0])
    utils.pickle_dump(y_valid, config["valid_set_path"][1])

    utils.pickle_dump(x_test, config["test_set_path"][0])
    utils.pickle_dump(y_test, config["test_set_path"][1])

    utils.pickle_dump(raw_dataset, config["dataset_cleaned_path"])