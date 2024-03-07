import os
import random
import pandas as pd
from datetime import datetime, timedelta

# Создаем папки, если они еще не существуют
if not os.path.exists('train'):
    os.makedirs('train')
if not os.path.exists('test'):
    os.makedirs('test')

# Генерируем данные для тренировочного и тестового наборов
def generate_data(start_date, end_date, num_points, noise_level):
    dates = [start_date + timedelta(days=i) for i in range(num_points)]
    values = [random.uniform(50, 100) for _ in range(num_points)]
    noisy_values = [v + random.uniform(-noise_level, noise_level) for v in values]
    return pd.DataFrame({'Date': dates, 'Value': noisy_values})

# Параметры генерации данных
train_data_params = [
    (datetime(2010, 1, 1), datetime(2011, 12, 31), 730, 0.5),
    (datetime(2012, 1, 1), datetime(2013, 12, 31), 731, 0.4),
    (datetime(2014, 1, 1), datetime(2015, 12, 31), 730, 0.6),
    (datetime(2016, 1, 1), datetime(2017, 12, 31), 731, 0.3),
    (datetime(2018, 1, 1), datetime(2019, 12, 31), 730, 0.7)
]

test_data_params = [
    (datetime(2020, 1, 1), datetime(2021, 12, 31), 731, 0.5),
    (datetime(2022, 1, 1), datetime(2023, 12, 31), 730, 0.4),
    (datetime(2024, 1, 1), datetime(2025, 12, 31), 731, 0.6)
]

# Создаем и сохраняем данные для тренировочного набора
for i, params in enumerate(train_data_params):
    start_date, end_date, num_points, noise_level = params
    df_train = generate_data(start_date, end_date, num_points, noise_level)
    df_train.to_csv(f'train/train_data_{i}.csv', index=False)

# Создаем и сохраняем данные для тестового набора
for i, params in enumerate(test_data_params):
    start_date, end_date, num_points, noise_level = params
    df_test = generate_data(start_date, end_date, num_points, noise_level)
    df_test.to_csv(f'test/test_data_{i}.csv', index=False)
