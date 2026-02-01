import hic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype
import Helper
from scipy.cluster.hierarchy import dendrogram

df_nat=pd.read_csv("dataIN/NatLocMovements.csv")
df_pop=pd.read_csv("dataIN/PopulationLoc.csv")

def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)


nan_replace_df(df_nat)
nan_replace_df(df_pop)

df_final=df_nat.merge(df_pop[['Siruta','Population','CountyCode']],on='Siruta')
coloana = 'CountyCode'
cerinta1=df_final.groupby(coloana)[['LiveBirths','Deceased','Population']].sum()
cerinta1['RSNA']=(cerinta1['LiveBirths']-cerinta1['Deceased'])*1000/cerinta1['Population']
cerinta1 = cerinta1.reset_index()
cerinta1[['CountyCode','RSNA']].to_csv("dataOUT/Cerinta1.csv",index=False)
print("Am terminat cerinta 1")

coloane=['Marriages','Deceased','DeceasedUnder1Year','Divorces','StillBirths','LiveBirths']
df_final['Rata Marriages'] = df_final['Marriages']*1000/df_final['Population']
df_final['Rata Deceased'] = df_final['Deceased']*1000/df_final['Population']
df_final['Rata DeceasedUnder1Year'] = df_final['DeceasedUnder1Year']*1000/df_final['Population']
df_final['Rata Divorces'] = df_final['Divorces']*1000/df_final['Population']
df_final['Rata StillBirths'] = df_final['StillBirths']*1000/df_final['Population']
df_final['Rata LiveBirths'] = df_final['LiveBirths']*1000/df_final['Population']

cerinta2 = pd.DataFrame()
for col in coloane:
    nume_rata = f'Rata {col}'
    idx=df_final.groupby('CountyCode')[nume_rata].idxmax()
    cerinta2[col]=df_final.loc[idx].set_index('CountyCode')['City']

cerinta2 = cerinta2.reset_index()
cerinta2.to_csv("dataOUT/Cerinta2.csv",index=False)

print("am terminat cerinta 2!")

# Helper.solve_hca('dataIN/DataSet_34.csv')

def citire_csv(nume_fisier, index_col=None):
    return pd.read_csv(nume_fisier, index_col=index_col)

def standardizare_df(t: pd.DataFrame, scal=True, ddof=0):
    x = t.values.astype(float)
    x_std = x - np.mean(x, axis=0)
    if scal:
        x_std = x_std / np.std(x, axis=0, ddof=ddof)
    return pd.DataFrame(x_std, index=t.index, columns=t.columns)

def salvare_df(t: pd.DataFrame, nume_fisier):
    t.to_csv(nume_fisier)
    return t

def get_thr(h):
    dist = h[:, 2]  # Get distances
    m = len(dist)
    diff = dist[1:] - dist[:m - 1]  # Calculate differences
    diff_max = np.argmax(diff)  # Find index of max difference
    # Return average of the two steps at the jump
    return (h[diff_max, 2] + h[diff_max + 1, 2]) / 2

def plot_ierarhie(h, etichete, thr, titlu="Dendrograma"):
    plt.figure(figsize=(12, 7))
    plt.title(titlu, fontsize=16)
    dendrogram(h, labels=etichete, leaf_rotation=45, color_threshold=thr)
    plt.axhline(thr, c='r', linestyle='--')
    plt.savefig("dataOUT/"+titlu+".png")


df_cluster = citire_csv("dataIN/DataSet_43.csv", index_col=0)
nan_replace_df(df_cluster)

# 2. Standardizare (Functia ta - CRITICA)
df_std = standardizare_df(df_cluster)
salvare_df(df_std, "dataOUT/Xstd.csv")  # Cerinta 3
print("[B] Xstd.csv salvat.")

# 3. Ierarhie (Asta e din scipy, nu e in functiile tale)
h = hic.linkage(df_std.values, method='ward', metric='euclidean')
print("Matricea ierarhica:\n", h)

# 4. Prag (Aici folosim functia get_thr adaugata manual mai sus)
thr = get_thr(h)
print("Prag (Threshold):", thr)

# 5. Grafic (Functia plot_ierarhie adaugata manual din setul tau grafic)
plot_ierarhie(h, df_cluster.index, thr)
print("[B] Dendrograma salvata.")

