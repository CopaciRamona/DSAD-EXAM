import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

df_coduri=pd.read_csv('dataIN/CoduriTariExtins.csv')
df_rata=pd.read_csv('dataIN/Rata.csv')

def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)



nan_replace_df(df_coduri)
nan_replace_df(df_rata)

df_final=df_coduri.merge(df_rata[['Three_Letter_Country_Code','FR','IM','LE','LEF','LKM','MMR','RS']],left_on='Country_Letter_code',right_on='Three_Letter_Country_Code')
valori=df_final[df_final['RS'] < df_final['RS'].mean()].sort_values(by='RS',ascending=False)
cerinta1=valori[['Three_Letter_Country_Code','Country','RS']]
cerinta1.to_csv('dataOUT/Cerinta1.csv',index=False)


coloane=list(df_rata.columns[2:])
df_lucru=df_final.set_index('Country_Letter_code')
cerinta2=df_lucru.groupby('Continent')[coloane].idxmax()
cerinta2.to_csv('dataOUT/Cerinta2.csv')

X_df = df_final.set_index('Country_Letter_code')[coloane]
X = X_df.values

# 2. STANDARDIZARE (Obligatorie la PCA)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
pca = PCA()
C = pca.fit_transform(X_std)          # Componentele (Scorurile)
alpha = pca.explained_variance_       # Valorile proprii (Alpha)
ratio = pca.explained_variance_ratio_ # Procentul de varianta explicata

# --- B.1 TABELUL VARIANTEI (Varianta.csv) ---
df_varianta = pd.DataFrame({
    'Varianta (Alpha)': alpha,
    'Varianta Cumulata': np.cumsum(alpha),
    'Procent Explicat': ratio * 100,
    'Procent Cumulat': np.cumsum(ratio) * 100
})
df_varianta.index = [f'Comp_{i+1}' for i in range(len(alpha))]
df_varianta.to_csv('dataOUT/Varianta.csv')
print(df_varianta.head())

# --- B.2 SCREE PLOT (Graficul Variantei) ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(alpha) + 1), alpha, 'ro-', label='Varianta (Alpha)')
# Criteriul Kaiser (Linia orizontala la 1)
plt.axhline(y=1, color='b', linestyle='--', label='Criteriul Kaiser')
plt.title("Scree Plot - Varianta Componentelor")
plt.xlabel("Componenta")
plt.ylabel("Valoare Proprie (Alpha)")
plt.legend()
plt.show() #

# --- B.3 CORELOGRAMA (Corelatii Factoriale) ---
loadings = pca.components_.T
corelatii = loadings * np.sqrt(alpha)
df_corelatii = pd.DataFrame(
    corelatii,
    index=coloane,
    columns=[f'C{i+1}' for i in range(len(alpha))]
)

plt.figure(figsize=(10, 8))
sns.heatmap(df_corelatii, annot=True, cmap='RdBu', vmin=-1, vmax=1)
plt.title("Corelograma (Relatia Variabile - Componente)")
plt.show() #


#CERINTA 3
valori = [0.9, 0.05, 0.8, 0.75, 0.5, 0.3, 0.2]
valori.sort(reverse=True)
total = sum(valori)
suma = 0
for i, x in enumerate(valori):
    suma += x
    procent = (suma / total) * 100
    if procent > 90:
        print(f"Ai nevoie de {i + 1} variabile.")
        break