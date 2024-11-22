from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd

def read_csv_file():
    Tk().withdraw()
    file = askopenfilename(filetypes=[("CSV Files", "*.csv")])
    read_columns =['cultivarName', 'alphaAcidsAverage', 'betaAcidsAverage', 'cohumuloneAverage', 'hopType']
    df = pd.read_csv(file, usecols=read_columns)
    return df