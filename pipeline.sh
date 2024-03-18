#!/bin/bash

# Запуск скрипта для создания данных
echo "Создание данных..."
python data_creation.py

# Запуск скрипта для предобработки данных
echo "Предобработка данных..."
python model_preprocessing.py

# Запуск скрипта для обучения модели
echo "Обучение модели..."
python model_preparation.py

# Запуск скрипта для тестирования модели
echo "Тестирование модели..."
python model_testing.py

echo "Готово."
