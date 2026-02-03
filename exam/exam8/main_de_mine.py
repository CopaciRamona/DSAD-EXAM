import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype


df_populatie=pd.read_csv('dataIN/PopulatieLocalitati.csv')
df_industrie=pd.read_csv('dataIN/Industrie.csv')

def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)


nan_replace_df(df_populatie)
nan_replace_df(df_industrie)

#Cerinta1
coloane=list(df_industrie.columns[2:])
df_final=df_industrie.merge(df_populatie[['Siruta','Populatie','Judet']],on='Siruta')
df_lucru=df_final.set_index('Siruta')
for col in coloane:
    df_lucru[col]=df_lucru[col]/df_lucru['Populatie']
cols=['Localitate'] + coloane

cerinta1=df_lucru[cols]
cerinta1.to_csv('dataOUT/Cerinta1.csv')

#Cerinta2
df_rezultat=df_final.groupby('Judet')[coloane].sum()
activitate=df_rezultat.idxmax(axis=1)
valoare_maxima=df_rezultat.max(axis=1)
cerinta2=pd.DataFrame({
    "Activitate":activitate,
    "Cifra Afacerii":valoare_maxima
})
cerinta2.to_csv('dataOUT/Cerinta2.csv')

