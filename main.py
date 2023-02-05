from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pickle
import numpy
import pandas
import gunicorn

class model(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

pickle_in = open("classifier.pkl", "rb")
cls = pickle.load(pickle_in)

myApp = FastAPI()

@myApp.get("/")
def homeFunction():
    return "Hello"

@myApp.post("/water_quality")
def getStudent(quer : model):
    query = quer.dict()
    parameters = [[query['ph'],query['Hardness'],query['Solids'],query['Chloramines'],query['Sulfate'],query['Conductivity'],query['Organic_carbon'],query['Trihalomethanes'],query['Turbidity']]]
    arr = numpy.array(parameters, dtype=float)
    columns = []
    for i in query.keys():
        columns.append(i)
    df = pandas.DataFrame(arr, columns=columns)
    output = cls.predict(df)
    if(output[0]==1):
        return "Safe to drink"
    else:
        return "Unsafe to drink"

@myApp.get("/gTinfo")
def get_info():
    return "This is a API to predict Water Quality"


#uvicorn.run(myApp)
