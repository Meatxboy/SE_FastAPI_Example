# -*- coding: utf-8 -*-
# Модель определяет позитивную, нейтральную или негативную эмоцию содержит в себе текст.

from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


class Item(BaseModel):
    text: str


app = FastAPI()
classifier = pipeline("sentiment-analysis")


@app.get("/")
def root():
    return {"FastApi service started!"}

@app.get('/how/')
def how():
    text = "Hello world!"
    res = classifier(text)
    acc = round(res[0]['score'] * 100, 2)

    return 'Например, я считаю, что фраза "Hello world!" позитивна на ' + str(acc) + '%'

@app.get("/{text}")
def get_params(text: str):
    return classifier(text)


@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)
