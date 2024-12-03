from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd

def read_csv_file():
    Tk().withdraw()
    file = askopenfilename(filetypes=[("CSV Files", "*.csv")])
    read_columns =['structureId', 'classification', 'macromoleculeType', 'residueCount', 'structureMolecularWeight', 'densityMatthews', 'densityPercentSol', 'phValue']
    df = pd.read_csv(file, usecols=read_columns)
    print("Data samples (first five rows)")
    print(df[:5])
    return df