## Цель:
Выразить FC слой через nn.Conv2d

Слой:  my_fc_layer = nn.Linear(3 * 12 * 12, 7)

## Решение
Решение задачи продемонстрировано на основе датасета MNIST.

Превый FC слой nn.Linear(3 * 12 * 12, 7), а так же следующий за ним nn.Linear(7, 10), были заменены сверточными слоями.

Первый слой был заменен на: nn.Conv2d(3, 7, kernel_size=12, stride=1)

Второй на: nn.Conv2d(7, 10, kernel_size=1, stride=1)

## Результат

Сеть с FC слоями:

![convnet](https://github.com/iliakhar/FcToConv/blob/master/netron_res/convnet.onnx.png)

Сеть без FC слоев:

![full_convnet](https://github.com/iliakhar/FcToConv/blob/master/netron_res/full_convnet.onnx.png)

График обучения сети с FC слоями:

![convnet_graph](https://github.com/iliakhar/FcToConv/blob/master/graph/Convnet_graph.png)

График обучения сети без FC слоев:

![full_convnet_graph](https://github.com/iliakhar/FcToConv/blob/master/graph/Full_Convnet_graph.png)

## Запуск
### Установка зависимостей для Windows
Перейти в корневую папку и запусть:

pip install -r requirements.txt

### Обучение

В командной строке произвести запуск main.py со следующими параметрами:

*python main.py train model_type number_of_epochs lr batch_size save_folder model_name*

*model_type* - conv (обучение сети с FC слоями), full_conv (обучение сети без FC слоев);

*model_name* - название сохраняемой модели без указания разрешения (сохраняется в форматах .ckpt и .onnx).

Пример:

*python main.py train conv 10 0.001 100 model convnet*

После обучения в папке graph появится график точности модели на каждой эпохе в процессе обучения. Название графика - Convent_graph или Full_Convent_graph, зависит от заданного model_type.

### Тестирование

В командной строке произвести запуск main.py со следующими параметрами:

*python main.py test model_type batch_size model_path*

*model_type* - conv (тестирование сети с FC слоями), full_conv (тестирование сети без FC слоев);

*model_path* - путь до модели с расширением .ckpt.

Пример:

python main.py test conv 100 model/convnet.ckpt
