import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

# Путь к папке с тренировочными данными
train_data_folder = 'train'

# Проходим по всем файлам с тренировочными данными и применяем предобработку
for file_name in os.listdir(train_data_folder):
    if file_name.endswith('.csv'):
        # Загружаем данные из файла
        df_train = pd.read_csv(os.path.join(train_data_folder, file_name))

        # Проверяем стационарность временного ряда
        result = adfuller(df_train['Value'])

        # Применяем предобработку (стандартизация)
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df_train[['Value']])
        df_train['Value'] = scaled_values

        # Сохраняем предобработанные данные в ту же папку с добавлением префикса 'preprocessed_'
        df_train.to_csv(os.path.join(train_data_folder, 'preprocessed_' + file_name), index=False)

        result = adfuller(df_train['Value'])
        print('Проверка стационарности', file_name)
        print('ADF статистика:', result[0])
        print('p-значение:', result[1])
        print('Критические значения:')
        for key, value in result[4].items():
            print(f'  {key}: {value}')
        print('\n')
