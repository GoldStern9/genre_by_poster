# genre_by_poster  
В проекте используется датасет с **Kaggle**: https://www.kaggle.com/raman77768/movie-classifier  

Классификация жанра фильма по его постеру, используя предобученную модель **ResNet-50** и **PyTorch**  
[про ResNet можно посмотреть тут](https://pytorch.org/hub/pytorch_vision_resnet/)  
****
### Файловая структура:
![Структура](https://github.com/GoldStern9/genre_by_poster/raw/main/visualisation/d_str.png)  
- **data**
  - **Images**: постеры
  - **preproc_res**: "очищенные" версии изначального датасета
  - **train.csv**: изначальный датасет
- **data_preprocessing**
  - **prep_utilities.py**: скрипт предобработки данных
  - **eda_and_preprocess.ipynb**: разбор предобработки и немного визуализации
  - **pipeline_prep.ini**: пайплайн преобработки
- **models**
  - **model.pth**: обученная модель
- **src**
  - **model_engine.py**: движок тренировки и тестирования модели
  - **model_usage.py**: предсказание по одному или группе постеров
  - **set_data.py**: класс, описывающий датасет
  - **train.py**: обучение модели
  - **metrics.py**: метрики
- **visualisation**: как есть, визуализация и несколько предсказаний  
### Распределение данных:
![Распределение](https://github.com/GoldStern9/genre_by_poster/raw/main/visualisation/genre_distr_bar.png)
### Loss по эпохе:
![Loss](https://github.com/GoldStern9/genre_by_poster/raw/main/visualisation/loss.png)
### Примеры классификации:  
***Пример 1:***
****
![P1](https://github.com/GoldStern9/genre_by_poster/raw/main/visualisation/test_genres_vis/predictions_2.jpg)  
***Пример 2:***
****  
![P2](https://github.com/GoldStern9/genre_by_poster/raw/main/visualisation/test_genres_vis/predictions_3.jpg)   
***Пример 3:***
****  
![P3](https://github.com/GoldStern9/genre_by_poster/raw/main/visualisation/test_genres_vis/predictions_9.jpg)
