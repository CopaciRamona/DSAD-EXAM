import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch


df_air=pd.read_csv('dataIN/AirQuality.csv', index_col=0)
df_country=pd.read_csv('dataIN/CountryContinents.csv',index_col=0)

def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)


nan_replace_df(df_air)
nan_replace_df(df_country)

coloane=list(df_air.columns[1:])
df_lucru = df_air.set_index('Country')[coloane]
maxim = df_lucru.idxmax()
df_rezultat = pd.DataFrame({
    'Indicator': maxim.index,
    'Country': maxim.values
})
df_rezultat.to_csv('dataOUT/Cerinta1.csv', index=False)
print('Gata Cerinta 1');


df_final = df_air.merge(df_country[['Continent']], left_index=True, right_index=True)
df_final = df_final.set_index('Country')
df_rezultat2 = df_final.groupby('Continent')[coloane].idxmax()
df_rezultat2.to_csv('dataOUT/Cerinta2.csv')
print("Gata Cerinta 2 (vectorizat și fără erori).")



# Pregatire date (Indicatorii) - Pastram indexul Country pentru a avea etichete corecte
coloane = list(df_air.columns[1:])
df_lucru = df_air.set_index('Country')[coloane]

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
sch.dendrogram(matrice, labels=df_air.index, leaf_rotation=90)
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
df_popt['Country'] = df_air['Country'].values
df_popt = df_popt.set_index('Country')

df_popt.to_csv("dataOUT/popt.csv")

