import pandas as pd
import numpy as np
import os
import logging
import onnxruntime as ort
import fire
from transformers import BertTokenizerFast

#Настраиваем систему логирования в файл
os.makedirs('./data/', exist_ok=True)
logging.basicConfig(level=logging.INFO, filename="./data/log_file.log", filemode="w", format="%(asctime)s %(levelname)s %(message)s")

LABELS = ['generated', 'human'] #Создаем список меток классов
id2label = {idx:label for idx, label in enumerate(LABELS)} #Создаем словарь для преобразования индексов в метки

def data_Setting(df):
    """
    Функция для настройки Dataset, чтобы он мог работать с моделью onnx
    """
    try:
        logging.info("Beggining of setting up the dataset")

        #Получаем текст из столбца 'text' и преобразуем его в список
        df_text = df['text'].tolist()

        #Ограничиваем количество текстов до 500
        df_text = df_text[:500]
        
        #Инициализируем токенизатор BERT
        tokenizer = BertTokenizerFast.from_pretrained('distilbert-base-cased')

        #Токенизируем тексты
        test_encodings = tokenizer(df_text, truncation=True, max_length=256, padding=True)

        #Получаем идентификаторы токенов и маску внимания
        input_ids = test_encodings['input_ids']
        attention_mask = test_encodings['attention_mask']

        logging.info("Setting up the dataset was successful")

        return input_ids, attention_mask
    except Exception:
        logging.error("Something went wrong when setting up the dataset", exc_info=True)


class My_TextClassifier_Model():
    def __init__(self, dataset = None):
        """
        Конструктор класса для инициализации модели с заданным путем датасета.
        """
        logging.info("Beginning of model initialization")

        self.dataset = dataset

        logging.info("Model initialization was successful")

    def predict(self):
        """
        Метод для использования обученной модели для предсказания того, кто написал модель - человек или нейросеть.
        """

        try:
            logging.info("Start predicting results on new data")

            #Получаем Dataset для предсказания того, кто написал модель - человек или нейросеть
            df = pd.read_csv(self.dataset)

            #Создаем результат с текстом сочинения
            result = pd.DataFrame()
            result['text'] = df['text']
            result = result[:500]

            #Настраиваем Dataset
            input_ids, attention_mask = data_Setting(df)

            # Загрузка модели ONNX
            session = ort.InferenceSession("./data/torch-model.onnx")
            
            # Выполнение предсказания с помощью ONNX модели
            logits = session.run(['logits'], {'input_ids': input_ids, 'attention_mask': attention_mask})
            probabilities = np.exp(logits[0]) / np.sum(np.exp(logits[0]), axis=1, keepdims=True)
            preds = np.argmax(probabilities, axis=1)
            result['pred_label'] = [id2label[x] for x in preds]

            #Сохраняем результат
            os.makedirs('./data/result/', exist_ok=True)
            result.to_csv('./data/result/results.csv', index=False)

            logging.info("Prediction of results using new data was successful")
        except Exception:
            logging.error("Something went wrong when predicting results from new data", exc_info=True)

if __name__ == '__main__':
    try:
        logging.info("Launching the application")

        # Запуск модели через командную строку с помощью Fire
        fire.Fire(My_TextClassifier_Model)

        logging.info("Shutting down the application")
    except Exception:
        logging.error("Something went wrong", exc_info=True)
