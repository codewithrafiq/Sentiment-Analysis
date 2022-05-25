import pickle
from fastapi import FastAPI, APIRouter, Request, Form
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.templating import Jinja2Templates

from fastapi.responses import HTMLResponse

# Sentiment Analysis using Python

app = FastAPI()
router = APIRouter()
templates = Jinja2Templates(directory="templates")

from pydantic import BaseModel


class Sentence(BaseModel):
    sentence: str


class TwtterSentimentAnalysis:
    def __init__(self):
        self.loaded_model = pickle.load(open('src/finalized_model.sav', 'rb'))
        self.loaded_tokenizer = pickle.load(
            open('src/finalized_tokenizer.sav', 'rb'))
        self.sentiment_label = {1: 'Negative', 0: 'Positive'}

    def predict(self, sentence):
        tw = self.loaded_tokenizer.texts_to_sequences([sentence])
        tw = pad_sequences(tw, maxlen=200)
        prediction = int(self.loaded_model.predict(tw).round().item())
        sentiment = self.sentiment_label[prediction]
        return sentiment


class IMDBSentimentAnalysis:
    def __init__(self):
        self.loaded_model = pickle.load(open('src/finalized_model.sav', 'rb'))
        self.loaded_tokenizer = pickle.load(
            open('src/finalized_tokenizer.sav', 'rb'))
        self.sentiment_label = {1: 'Negative', 0: 'Positive'}

    def predict(self, sentence):
        tw = self.loaded_tokenizer.texts_to_sequences([sentence])
        tw = pad_sequences(tw, maxlen=200)
        prediction = int(self.loaded_model.predict(tw).round().item())
        sentiment = self.sentiment_label[prediction]
        return sentiment


@router.post('/predict')
async def predict(sentence: Sentence):
    sa = TwtterSentimentAnalysis()
    return {'prediction': sa.predict(sentence.sentence), 'sentence': sentence.sentence}

@router.post('/imdb_predict')
async def predict(sentence: Sentence):
    sa = IMDBSentimentAnalysis()
    return {'prediction': sa.predict(sentence.sentence), 'sentence': sentence.sentence}


@router.get('/', response_class=HTMLResponse)
async def index(request: Request,):
    return templates.TemplateResponse('index.html', {'request': request})


app.include_router(router)




import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)