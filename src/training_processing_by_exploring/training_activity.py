import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

import pandas as pd
import sys
from regression_models import regression_model
import os

doc_config = open(sys.argv[1], 'r')

df_data = pd.read_csv(doc_config.readline().replace("\n", ""))
name_export = doc_config.readline().replace("\n", "")
iteration = int(doc_config.readline().replace("\n", ""))

doc_config.close()

response = df_data["activity"]
df_to_train = df_data.drop(columns=['expression', "activity"])

X_train, X_test, y_train, y_test = train_test_split(df_to_train, response, random_state=iteration, test_size=0.3)

regx_instance = regression_model(
    X_train, 
    y_train, 
    X_test, 
    y_test
)

regx_instance.make_exploration()

if regx_instance.status == "OK":
    regx_instance.response_training['iteration'] = iteration
    regx_instance.response_training.to_csv(name_export, index=False)
