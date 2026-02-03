import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)

dp_global=pd.read_csv('dataIN/GlobalIndicatorsPerCapita_2021.csv')#si index_col=0
dp_country=pd.read_csv('dataIN/CountryContinents.csv')#si index_col=0

nan_replace_df(dp_global)
nan_replace_df(dp_country)

dp_final=dp_global.merge(dp_country[['CountryID','Continent']],left_on='CountryId',right_on='CountryID')
#dp_final=dp_global.merge(dp_country,right_index=True,left_index=True)
coloane = ['AgrHuntForFish', 'Construction', 'Manufacturing',
          'MiningManUt', 'TradeT', 'TransportComm', 'Other']
dp_final['Valoare Adaugata'] = dp_final[coloane].sum(axis=1)
cerinta1=dp_final[['CountryID','Country','Valoare Adaugata']]
cerinta1.to_csv('dataOUT/Cerinta1.csv',index=False)

print("Cerinta 1 a fost facuta!");

#INCEPUTCERINTA2

coloane_cerinta2=list(dp_global.columns[2:])
grupa=dp_final.groupby('Continent')[coloane_cerinta2]
cerinta2=grupa.apply(lambda x : x.std(ddof=1)/x.mean())
cerinta2.to_csv('dataOUT/Cerinta2.csv')

print("Am rezolvat cerinta 2!");


valori_numerice = dp_final[coloane_cerinta2].values
indicatori_standardizati = StandardScaler().fit_transform(valori_numerice)
pca= PCA()
c= pca.fit_transform(indicatori_standardizati)

n,m = indicatori_standardizati.shape
variatii = pca.explained_variance_
alpha = variatii * (n-1)/n

print('explained_variance: ', variatii)
print('alpha', alpha)

dp_final.set_index('Country', inplace=True)

# ------------CERINTA B.2.-------------
scoruri = pd.DataFrame(c, index= dp_global.index, columns=[f"CP{i}" for i in range(1, c.shape[1]+1)])
scoruri.to_csv('./dataOUT/scoruri.csv', index=True)


# ------------CERINTA B.3.-------------
plt.figure(figsize=(8,6))

# Scatter plot: Axa X = Coloana 0 (CP1), Axa Y = Coloana 1 (CP2)
plt.scatter(c[:, 0], c[:, 1], c='b', alpha=0.6)

# ETICHETARE PUNCTE (Foarte important pentru punctaj maxim!)
# Parcurgem fiecare punct și îi punem numele țării
tari = dp_final.index # Sau dp_final['Country'] dacă nu ai setat indexul
for i in range(len(tari)):
    plt.text(c[i, 0], c[i, 1], tari[i], fontsize=8)

# Detalii grafic
plt.xlabel(f"CP1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
plt.ylabel(f"CP2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
plt.title("Scoruri in primele 2 axe principale (CP1, CP2)")
plt.axhline(0, c='k', linestyle='--') # Linia orizontală prin 0
plt.axvline(0, c='k', linestyle='--') # Linia verticală prin 0
plt.grid()

plt.show()
print(">>> [B.3] Grafic afișat.")

#Cerinta C
df_g20=pd.read_csv('dataIN/g20.csv',index_col=0)
nan_replace_df(df_g20)
comunalitati=(df_g20 ** 2).sum(axis=1)
psi=1-comunalitati
print("Raspunsul e:", int(np.argmax(psi.values)+1))