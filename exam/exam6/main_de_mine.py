import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df_coduri=pd.read_csv('dataIN/CoduriTariExtins.csv')
df_mortalitati=pd.read_csv('dataIN/Mortalitate.csv')

def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)


nan_replace_df(df_coduri)
nan_replace_df(df_mortalitati)

#CERINTA1
df_rezultat=df_mortalitati[df_mortalitati['RS']<0]
cerinta1=df_rezultat[['Tara','RS']]
cerinta1.to_csv('dataOUT/Cerinta1.csv',index=False)

#CERINTA2
df_final=df_mortalitati.merge(df_coduri[['Tari','Continent']],left_on='Tara',right_on='Tari')
coloane=list(df_mortalitati.columns[1:])
cerinta2=df_final.groupby('Continent')[coloane].mean()
cerinta2.to_csv('dataOUT/Cerinta2.csv')

# valori_numerice = df_final[coloane].values
#
# # ==============================================================================
# #  REZOLVARE SUBIECT B (PCA) - CODUL TAU
# # ==============================================================================
#
# # 1. Standardizare
# scaler = StandardScaler()
# indicatori_standardizati = scaler.fit_transform(valori_numerice)
#
# # 2. Model PCA
# pca = PCA()
# c = pca.fit_transform(indicatori_standardizati)
#
# # 3. Varianta si Alpha
# n, m = indicatori_standardizati.shape
# variatii = pca.explained_variance_
# alpha = variatii * (n-1)/n
#
# print('explained_variance (Lambda): ', variatii)
# print('alpha (Varianta Populatie): ', alpha)
#
# # ------------ CERINTA B.2 (Salvare Scoruri) -------------
# # Folosim indexul din df_lucru pentru a nu avea erori de lungime
# scoruri = pd.DataFrame(
#     c,
#     index=df_final.index,
#     columns=[f"CP{i}" for i in range(1, c.shape[1]+1)]
# )
# scoruri.to_csv('./dataOUT/scoruri.csv', index=True)
# print("✅ B.2 Scoruri salvate in dataOUT/scoruri.csv")
#
#
# # ------------ CERINTA B.3 (Grafic) -------------
# plt.figure(figsize=(10, 7))
#
# # Scatter plot: CP1 vs CP2
# plt.scatter(c[:, 0], c[:, 1], c='dodgerblue', alpha=0.6, edgecolors='k')
#
# # ETICHETARE PUNCTE (Codul tau)
# tari = df_final.index
# for i in range(len(tari)):
#     plt.text(c[i, 0], c[i, 1], tari[i], fontsize=8, alpha=0.8)
#
# # Detalii grafic
# var_ratio = pca.explained_variance_ratio_
# plt.xlabel(f"CP1 ({var_ratio[0]*100:.2f}%)")
# plt.ylabel(f"CP2 ({var_ratio[1]*100:.2f}%)")
# plt.title("Scoruri in primele 2 axe principale (CP1, CP2)")
# plt.axhline(0, c='k', linestyle='--', alpha=0.5)
# plt.axvline(0, c='k', linestyle='--', alpha=0.5)
# plt.grid(True, alpha=0.3)
#
# plt.show()
# print(">>> [B.3] Grafic afișat.")

#CerintaB 1
valori_numerice = df_final[coloane].values
indicatori_standardizati = StandardScaler().fit_transform(valori_numerice)
pca = PCA()
c = pca.fit_transform(indicatori_standardizati)
n, m = indicatori_standardizati.shape
variatii = pca.explained_variance_
alpha = variatii * (n-1)/n

# --- AFISARE CERINTA B.1 ---
print('explained_variance (Sample): ', variatii)
print('alpha (Population): ', alpha)


# --- CERINTA B.2 (Scoruri) ---
scoruri = pd.DataFrame(c, index=df_final.index, columns=[f"CP{i}" for i in range(1, c.shape[1]+1)])
scoruri.to_csv('./dataOUT/scoruri.csv', index=True)



# --- CERINTA B.3 (Grafic) ---
plt.figure(figsize=(10, 7))
plt.scatter(c[:, 0], c[:, 1], c='b', alpha=0.6)
tari = df_final.index
for i in range(len(tari)):
    plt.text(c[i, 0], c[i, 1], tari[i], fontsize=8)
plt.xlabel(f"CP1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
plt.ylabel(f"CP2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
plt.title("Scoruri in primele 2 axe principale (CP1, CP2)")
plt.axhline(0, c='k', linestyle='--')
plt.axvline(0, c='k', linestyle='--')
plt.grid()
plt.show()


