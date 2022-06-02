# workplaceDetector
Тестовое задание на бинарную классификацию изображений.

## Подготовка данных
В качестве данных было предоставлено видео. Необходимо вырезать область, заданную координатами, из каждого 15 кадра и рескейльнуть до  размера 119x119. Скрипт - [video_data_extractor.py](video_data_extractor.py).

Затем в ручном режиме изображения раскинуты по двум папкам True и False, в зависимости, есть ли человек на рабочем месте. После чего скриптом [validation_set_generator.py](validation_set_generator.py) по 10% случайно выбранных изображений из каждой папки вынесены в валидационный набор.

## Нейронная сеть и обучение
В условии задания было указано, что чем меньше сеть, тем лучше. Поэтому моя нейронная сеть состоит из 3 слоев: сверточный в 20 слоев с ядром 20x20, maxpool с ядром 4x4 и линейный с 2 выходными нодами. Активации - ReLU, функция потерь - кросс-энтропия, оптимизатор - стохастический градиентный спуск. Просчитал 40 эпох. Скрипт обучения - [workplace_classifier_train.py](workplace_classifier_train.py).

## Метрики
Во время обучения после каждой эпохи фиксировались среднее значение функции потерь кросс-энтропии и confusion matrix (порог 0.5) для тренировочного и валидационного наборов. Результаты - [metrics.csv](metrics.csv). *В этом файле нумерация эпох с 0, во всех остальных местах я использую нумерацию с 1.*

В скрипте [metrics_analyzer.py](metrics_analyzer.py) расчет метрик в зависимости от эпохи.

### Функция потерь кросс-энтропии
Лучшая эпоха - 19 (33.3763).
![CE_losses](https://user-images.githubusercontent.com/25753000/171667668-016b6d0b-4f0c-4069-acc5-1333a558e410.png)

### F1 мера и другие метрики Confusion Matrix с порогом 0.5
Лучшая эпоха - 17 (0.9931).
![CM_analysis](https://user-images.githubusercontent.com/25753000/171667832-0426adf2-7332-41a9-b36c-fc0ce43b0e82.png)

Результаты расчетов в файле [metrics_calculated.csv](metrics_calculated.csv).

## ROC
Для лучших двух эпох ([17](savestates/17) и [19](savestates/19)) скриптом [ROC.py](ROC.py) для кадого порога от 0 до 1 с шагом 0.01 на валидационном наборе рассчитаны F1 мера, True Positive Rate и False Positive Rate. Значения в файле [ROC.csv](ROC.csv). Значения вероятностей принадлежности к классу 1, выдаваемые нейросетями, представлены в файле [17_19_validation.csv](17_19_validation.csv).

Скриптом [ROC_analyzer.py](ROC_analyzer.py) вычисленны площади под графиком:
- ROC_AUC_17 = 0.9975
- ROC_AUC_19 = 0.9970

### График ROC
![ROC](https://user-images.githubusercontent.com/25753000/171674358-efbb5ec5-c0d0-4a42-b40f-f5bba88ea11d.png)
##### Увеличенная область у точки (0;1)
![ROC_zoomed](https://user-images.githubusercontent.com/25753000/171674507-d2e40211-8ebf-4233-81dd-195dc41ffa35.png)

### F1 мера
![F1](https://user-images.githubusercontent.com/25753000/171674826-7a860fb3-6b37-41b3-831d-638b5885adb9.png)

## Интересное наблюдение
Нейросети до эпохи 17 включительно на позитивном результате на обеих нодах почти везде выдавали по 0, и, соответственно, оценивали вероятность в 0.5. На 18 эпохе все поменялось, и на второй ноде появилось положительное число, что привело к сильному падению в метриках, но было исправлено уже на 19 эпохе.
