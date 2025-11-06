import pandas as pd
from src.data import load_dataset, split_xy
from src.model import build_pipeline

def test_pipeline_fits():
    df = pd.DataFrame({
        "day":[1,2,3,4],
        "pressure":[1025.9,1022.0,1020.0,1018.0],
        "maxtemp":[19.9,21.7,22.1,18.0],
        "temparature":[18.3,18.9,19.2,17.5],
        "mintemp":[16.8,17.2,16.5,15.9],
        "dewpoint":[13.1,15.6,14.0,12.5],
        "humidity":[72,81,76,70],
        "cloud":[49,83,60,40],
        "sunshine":[9.3,0.6,5.0,7.0],
        "winddirection":[80,50,120,210],
        "windspeed":[26.3,15.3,10.0,20.0],
        "rainfall":["yes","no","yes","no"]
    })
    df.to_csv("tmp.csv", index=False)
    df2 = load_dataset("tmp.csv")
    X, y = split_xy(df2)
    pipe = build_pipeline(X.columns)
    pipe.fit(X, y)
    assert len(pipe.predict(X)) == len(y)
