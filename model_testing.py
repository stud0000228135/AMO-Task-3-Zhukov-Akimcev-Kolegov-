import os
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Путь к файлу с обученной моделью
model_file = 'train/trained_model.pkl'

# Путь к папке с тестовыми данными
test_data_folder = 'test'

# Загружаем обученную модель из файла
model = joblib.load(model_file)

# Проходим по всем файлам с тестовыми данными и тестируем модель
for file_name in os.listdir(test_data_folder):
    if file_name.endswith('.csv'):
        # Загружаем данные из файла
        df_test = pd.read_csv(os.path.join(test_data_folder, file_name))

        # Предсказываем значения с помощью модели ARIMA
        # Важно: для ARIMA нужно определить предыдущие точки временного ряда,
        # но в этом примере мы просто будем делать предсказания на основе предыдущих значений Value
        # Это не самый точный способ, но для целей примера он подходит
        predictions = model.forecast(steps=len(df_test))  # Прогнозируем столько же шагов, сколько и в тестовом наборе

        # Выводим результаты предсказаний
        print(f"Predictions for {file_name}: {predictions}")

        # Оцениваем качество модели
        true_values = df_test['Value']
        mse = mean_squared_error(true_values, predictions)
        mae = mean_absolute_error(true_values, predictions)
        print(f'MSE for {file_name}: {mse}')
        print(f'MAE for {file_name}: {mae}')
        print('\n')
