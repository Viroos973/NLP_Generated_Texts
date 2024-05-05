FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install ./dist/nlp_generated_texts-0.1.0-py3-none-any.whl

CMD ["python3", "./nlp_generated_texts/flask_app.py"]