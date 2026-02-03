import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype
from plotly.figure_factory._dendrogram import sch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

coloane=list(df_alcohol.columns[2:])

df_lucru=df_alcohol.set_index('Code')
df_lucru['Consum Mediu']=df_lucru[coloane].mean(axis=1)
cerinta1=df_lucru[['Country','Consum Mediu']].sort_values(by='Consum Mediu',ascending=False)
cerinta1.to_csv('dataOUT/Cerinta1.csv')
print('Am terminat cerinta 1!')



df_final=df_alcohol.merge(df_coduri[['Country','Continent']],on='Country')
df_lucru_2=df_final.groupby('Continent')[coloane].mean()
df_rez=df_lucru_2.idxmax(axis=1)
cerinta2= pd.DataFrame({
    'Continent_Name': df_rez.index,
    'Anul': df_rez.values
})

cerinta2.to_csv('dataOUT/Cetinta2.csv',index=False)
print('Am termina cerinta 2!')

#Cerinta B1
#Matricea ierarhie cu ward
f_lucru = df_final.set_index('Country')
X = f_lucru[coloane].values
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
matrice = sch.linkage(X_std, method='ward')
print("Matricea Ierarhie (Ward):\n", matrice)

#Componenta partitiei din 5 clusteri
k_ales = 5
labels_k = sch.fcluster(matrice, t=k_ales, criterion='maxclust')
df_export = pd.DataFrame()
df_export['Code'] = df_final['Code'].values       # Sau 'Three_Letter_Country_Code'
df_export['Country'] = f_lucru.index.values      # Numele tarii
df_export['Cluster'] = labels_k                   # Clusterul calculat

df_export.to_csv("dataOUT/p4.csv", index=False)

#Plotul partitiei din 4 clusteri, folosim PCA cu doua coomponenete clan
# 2. Desenam (Scatter plot)
pca = PCA(n_components=2)
C = pca.fit_transform(X_std)
plt.figure(figsize=(10, 7))
plt.scatter(C[:, 0], C[:, 1], c=labels_k, cmap='rainbow', s=70, edgecolors='k')
plt.title(f"Partitia in {k_ales} clusteri pe axele principale")
plt.xlabel("Componenta Principala 1")
plt.ylabel("Componenta Principala 2")
plt.grid(True)
plt.show() #


#CERINTA3
df_a = pd.read_csv('dataIN/a.csv', header=None)
A = df_a.values
x = np.array([3, 1, 2, 1, 4])
scoruri = np.dot(x, A)
print(scoruri)