import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype
from plotly.figure_factory._dendrogram import sch
from sklearn.preprocessing import StandardScaler

df_alcohol=pd.read_csv('dataIN/alcohol.csv')
df_coduri=pd.read_csv('dataIN/CoduriTariExtins.csv')

def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)


nan_replace_df(df_alcohol)
nan_replace_df(df_coduri)

#CERINTA2
coloane=list(df_alcohol.columns[2:])
df_rezultat=df_alcohol.set_index('Entity')
df_rezultat['Media']=df_rezultat[coloane].mean(axis=1)
cerinta1=df_rezultat[['Code','Media']]
cerinta1.to_csv('dataOUT/Cerinta1.csv',index=False)



#CERINTA2

df_final=df_alcohol.merge(df_coduri[['Tari','Continent']],left_on='Entity',right_on='Tari')
rezultat=df_final.groupby('Continent')[coloane].mean()
grup=rezultat.idxmax(axis=1)
cerinta2=pd.DataFrame({
    "Continent_Name" : grup.index,
    "Anul": grup.values
})

cerinta2.to_csv('dataOUT/Cerinta2.csv',index=False)

df_lucru = df_alcohol.set_index('Entity')[coloane]

X = df_lucru.values
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
matrice = sch.linkage(X_std, method='ward')
print("Matricea:\n", matrice)

#  Calcul k_optim - Elbow
dist_agregare = matrice[:, 2]
difere_dist = np.diff(dist_agregare)
k_optim = len(dist_agregare) - np.argmax(difere_dist)
print("Numar optim clusteri (Elbow):", k_optim)

# Dendrograma cu linia de prag
prag = (dist_agregare[len(dist_agregare) - k_optim] + dist_agregare[len(dist_agregare) - k_optim - 1]) / 2
plt.figure(figsize=(10, 6))
sch.dendrogram(matrice, labels=df_alcohol.index, leaf_rotation=90)
plt.axhline(y=prag, color='r', linestyle='--', label=f'Prag optim (k={k_optim})')
plt.title("Dendrograma (Ward)")
plt.legend()
plt.tight_layout()
plt.show()

# --- C. Partitie si SALVARE --
k_ales = k_optim
labels_k = sch.fcluster(matrice, t=k_ales, criterion='maxclust')
df_popt = pd.DataFrame({
    'Cluster': labels_k
})
df_popt['Entity'] = df_alcohol['Entity'].values
df_popt = df_popt.set_index('Entity')

df_popt.to_csv("dataOUT/popt.csv")

