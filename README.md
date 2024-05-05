# Обо мне
ФИО: Елисеев Юрий Германович<br/>
Группа: 972201
# Как пользоваться с docker
- Скачайте проект с репозитория
- Скачайте torch-model.onnx с [googl disk](https://drive.google.com/file/d/1uE7Uh6oyC4QbGIot9A64YGXe-5UHU0dq/view?usp=sharing)
- Создайте на вашем диске "С" директорию "dataNLP", в которую поместите ваши torch-model.onnx и test.csv
- Откройте терминал
    - Если у вас Linux или Mac, то нет проблем
    - Если у вас Windows, то используйте WSL2
- Зайдите в корневой файл установленного репозитория
- Введите `foo@bar:~$ docker compose up`
- Для предсказаний модели отправьте POST запрос http://0.0.0.0:5000/predict?dataset=/app/data/path/to/test.csv
  - Вместо `path/to/test.cs` укажите путь к `test.csv` из директории `/c/dataNLP/`. Например: если `test.csv` лежит у вас на пути `/c/dataNLP/test.csv`, то укажите путь `/app/data/test.cs`
  - После выполнения запроса, результат предсказания появится в папке `/c/dataNLP/`
- В папке `/c/dataNLP/` появится файл `log_file.log` внутри которого можно будет увидеть как проходило предсказние модели
# Как пользоваться без docker
- Скачайте проект с репозитория
- Скачайте torch-model.onnx с [googl disk](https://drive.google.com/file/d/1uE7Uh6oyC4QbGIot9A64YGXe-5UHU0dq/view?usp=sharing)
- Поместите torch-model.onnx в директории `./data/` внутри корневого файла установленного репозитория
- Создайте директорию, в которую поместите ваш test.csv
- Откройте терминал
- Зайдите в корневой файл установленного репозитория
- Введите `foo@bar:~$ pip install ./dist/nlp_generated_texts-0.1.0-py3-none-any.whl`
- Предсказать модель можно через flask
  - Запустите файл flask_app.py
  - Для предсказаний модели отправьте POST запрос http://0.0.0.0:5000/predict?dataset=/path/to/test.csv
    - Вместо `path/to/test.cs` укажите путь к `test.csv`. Например: если `test.csv` лежит у вас на пути `/c/data/test.csv`, то укажите путь `/c/data/test.csv`
    - После выполнения запроса, результат предсказания появится в папке `./data/` внутри корневого файла установленного репозитория
- Предсказать модель можно через командную строку
  - Перед предсказанием модели необходимо перейти в папку `nlp_generated_texts`
  - Для предсказаний модели введите в командную строку `foo@bar:~$ python model.py predict --dataset=/path/to/test.csv`, `/path/to/test.csv` заменять аналогично flask
- В папке `./data/` появится файл `log_file.log` внутри которого можно будет увидеть как проходило обучение и предсказние и среднюю точность модели
# Чем пользовался
- [python = "3.10.12"](https://www.python.org/)
- WSL2
- Пакеты
  - [fire = "0.5.0"](https://google.github.io/python-fire/guide/)
  - [numpy = "1.26.4"](https://numpy.org/)
  - [pandas = "2.2.0"](https://khashtamov.com/ru/pandas-introduction/)
  - [Flask = "3.0.2"](https://flask.palletsprojects.com/en/3.0.x/)
  - [onnxruntime = "1.17.3"](https://onnxruntime.ai/)
  - [transformers = "4.39.3"](https://huggingface.co/docs/transformers/index)_
  - [poetry](https://python-poetry.org/docs/)
- Расширения
  - Docker
  - Pylance
  - Python
  - Python Debugger
- IDE
  - [VS code](https://code.visualstudio.com/)
  - [Anaconda](https://www.anaconda.com/)
  - [Jupyter notebook](https://jupyter.org/)
