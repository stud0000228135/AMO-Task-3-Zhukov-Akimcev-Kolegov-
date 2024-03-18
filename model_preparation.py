import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib

# Путь к папке с тренировочными данными
train_data_folder = 'train'

# Проходим по всем предобработанным файлам с тренировочными данными и создаем и обучаем модель
for file_name in os.listdir(train_data_folder):
    if file_name.startswith('preprocessed_'):
        # Загружаем предобработанные данные из файла
        df_train = pd.read_csv(os.path.join(train_data_folder, file_name))

        # Создаем и обучаем модель ARIMA
        model = ARIMA(df_train['Value'], order=(1, 1, 1))
        model_fit = model.fit()

        # Сохраняем обученную модель в файл
        joblib.dump(model_fit, os.path.join(train_data_folder, 'trained_model.pkl'))

        print(model_fit.summary())
        print('\n')
