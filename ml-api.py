from fastapi import FastAPI
from pydantic import BaseModel 
import pickle
import pandas as pd 

app = FastAPI()

class features(BaseModel):

    duration: float
    cast_total_facebook_likes: int
    budget: float
    imdb_score: float

# {
#   "duration": 120,
#   "cast_total_facebook_likes": 20000,
#   "budget": 15000000,
#   "imdb_score": 9.5
# }

with open('modelo-ml.pkl', 'rb') as f:

    modelo = pickle.load(f)


@app.post('/')
async def ml_endpoint(item: features):
    
    df = pd.DataFrame([item.dict().values()], columns= item.dict().values())

    prediccion = modelo.predict(df)

    return {"prediccion": int(prediccion)}
    


