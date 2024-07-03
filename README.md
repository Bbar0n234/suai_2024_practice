# MiniCPM-V-2.0 Finetuning
[Исходный репозиторий с моделью](https://github.com/OpenBMB/MiniCPM-V/tree/main)

Разработка данного репозитория велась в рамках летней Учебной практики 2024 года студентами группы 3234к.

Данный код адаптирован под обучение на очень небольших вычислительных мощностях: разработчики этой нейросети не упоминали обучение меньше чем на двух видеокартах **A100** (80G).
Мы же смогли произвести запуск обучения и последующего тестирования модели на бесплатном графическом ускорителе **T4 GPU**, и с ограничением по времени порядка двух-трёх часов.

Обучение не привело к сильному результату, однако мы всё-таки смогли провести полноценный цикл обучения, при этом не потратив абсолютно никаких средств на вычислительные ресурсы и разметку данных. 

### Файлы
 - **data/** - директория со всеми данными для обучения

   - **data/source_data/** - [исходные данные](https://www.kaggle.com/datasets/itsmariodias/vqa-validation-dataset)

   - **data/preprocessed_data/** - подготовленные данные (в нужном нам виде)
 - **scripts/** - вспомогательные функции
   - **scripts/preparation.py** - для подготовки данных
   - **scripts/finetuning.py** - для построения графиков Loss'а после обучения
 - **fine_tuning.ipynb** - Ноутбук с запуском самого Fine-tuning'а. Чтобы запустить обучение, нужно: 
   1. Поменять в соответствующих ячейках нужные пути к данным (в случае необходимости)
   2. Выполнить весь код последовательно
   3. Также нужно прочитать комментарии в ячейках, и при необходимости раскомментировать нужный код (в случае если используете Google Colab)
 - **inference.ipynb** - Ноутбук с Inference'ом модели(после Fine-tuning'а). Для его работы необходимо: 
   1. Запустить весь код в ячейка последовательно
   2. При необходимости раскомментировать нужный код (в случае если используете Google Colab)


