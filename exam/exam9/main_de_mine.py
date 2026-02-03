import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sns
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

df_emis=pd.read_csv('dataIN/emmissions.csv')
df_pop=pd.read_csv('dataIN/PopulatieEuropa.csv')

def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)


nan_replace_df(df_emis)
nan_replace_df(df_pop)

#Cerinta1
coloane=list(df_emis.columns[2:])
df_rezultat=df_emis.set_index('ThreeLetterCountryCode')
df_rezultat["Emisii"]=df_rezultat[coloane].sum(axis=1)
rezultat=df_rezultat[['Country','Emisii']]
rezultat.to_csv('dataOUT/Cerinta1.csv')


#cerinta 2
df_merge= df_emis.merge(df_pop[["ThreeLetterCountryCode","Region","Population"]],on="ThreeLetterCountryCode")
for col in coloane:
    df_merge[col] = (df_merge[col] / df_merge['Population']) * 100000

result = df_merge[['Region'] + coloane].drop_duplicates(subset="Region")
result.to_csv('dataOUT/Cerinta2.csv', index=False)

df_prod = pd.read_csv('dataIN/ElectricityProduction.csv')
nan_replace_df(df_prod)

# 2. Facem MERGE intre Productie (X) si Emisii (Y) folosind codul tau 'df_merge' logic
# Avem nevoie de un dataframe comun cu toate datele aliniate
df_cca = df_prod.merge(df_emis, on='ThreeLetterCountryCode')

# Setam indexul pe codul tarii (pentru etichete la grafice)
df_cca.set_index('ThreeLetterCountryCode', inplace=True)

# 3. Definim coloanele pentru cele doua seturi
# Set X: Productia (Verifica daca numele sunt Coal, Oil etc in fisier)
cols_x = ['Coal', 'Oil', 'Gas', 'Nuclear', 'Hydro', 'Renewable', 'Other']
# Set Y: Emisiile (Folosim lista 'coloane' definita de tine mai sus)
cols_y = coloane

X_vals = df_cca[cols_x].values
Y_vals = df_cca[cols_y].values

# 4. Standardizare (StandardScaler din sklearn)
scaler = StandardScaler()
X_std = scaler.fit_transform(X_vals)
Y_std = scaler.fit_transform(Y_vals)

# --- A. Modelul CCA si Scoruri Canonice ---
n_comp = min(X_std.shape[1], Y_std.shape[1])
cca = CCA(n_components=n_comp)
cca.fit(X_std, Y_std)
Z, U = cca.transform(X_std, Y_std)

# Salvare Scoruri (Z pt X, U pt Y)
pd.DataFrame(Z, index=df_cca.index, columns=[f'Z{i+1}' for i in range(n_comp)]).to_csv('dataOUT/z.csv')
pd.DataFrame(U, index=df_cca.index, columns=[f'U{i+1}' for i in range(n_comp)]).to_csv('dataOUT/u.csv')
print("✅ B.1 Scorurile Z si U salvate.")

# --- B. Corelatii Canonice ---
cor_can = [np.corrcoef(Z[:, i], U[:, i])[0, 1] for i in range(n_comp)]
cor_can = np.array(cor_can)
print("Corelatii canonice (r):", cor_can)
pd.DataFrame(cor_can, columns=['Corelatie']).to_csv('dataOUT/r.csv', index=False)

# --- C. Testul Bartlett (Functia din Cheat Sheet) ---
def test_bartlett(r, n, p, q):
    m = n - 1 - 0.5 * (p + q + 1)
    chi_sq = -m * np.log(np.prod(1 - r**2))
    df_bartlett = p * q
    return chi_sq, df_bartlett

chi_val, df_val = test_bartlett(cor_can, len(df_cca), X_std.shape[1], Y_std.shape[1])
print(f"Test Bartlett Global: Chi_sq={chi_val:.4f}, df={df_val}")

# --- D. Structura (Corelatii variabile - scoruri) ---
# Corelatii X cu Z
cor_x_z = np.corrcoef(X_std, Z, rowvar=False)[:X_std.shape[1], X_std.shape[1]:]
# Corelatii Y cu U
cor_y_u = np.corrcoef(Y_std, U, rowvar=False)[:Y_std.shape[1], Y_std.shape[1]:]

# --- E. Cercul Corelatiilor (Biplot) ---
plt.figure(figsize=(8, 8))
plt.scatter(cor_x_z[:, 0], cor_x_z[:, 1], c='r', label='Set X (Productie)')
plt.scatter(cor_y_u[:, 0], cor_y_u[:, 1], c='b', label='Set Y (Emisii)')

for i, txt in enumerate(cols_x):
    plt.text(cor_x_z[i,0], cor_x_z[i,1], txt, color='red')
for i, txt in enumerate(cols_y):
    plt.text(cor_y_u[i,0], cor_y_u[i,1], txt, color='blue')

plt.title("Cercul Corelatiilor Canonice")
plt.xlabel("Variabila Canonica 1")
plt.ylabel("Variabila Canonica 2")
plt.axhline(0, c='k', ls='--'); plt.axvline(0, c='k', ls='--')
plt.legend()
plt.show() #

# --- F. Corelograma (Heatmap) ---
plt.figure(figsize=(10, 6))
# Afisam relatia dintre Productie (X) si primele 2 variabile canonice
df_heatmap = pd.DataFrame(cor_x_z[:, :2], index=cols_x, columns=['Radacina 1', 'Radacina 2'])
sns.heatmap(df_heatmap, annot=True, cmap='RdBu', vmin=-1, vmax=1)
plt.title("Corelatii Variabile X - Scoruri Canonice")
plt.show()

# --- G. Plot Instante (Z1 vs U1) ---
plt.figure(figsize=(8, 6))
plt.scatter(Z[:, 0], U[:, 0], c='green', alpha=0.5)

# Etichete Tari
for i, txt in enumerate(df_cca.index):
    plt.text(Z[i, 0], U[i, 0], txt, fontsize=8)

plt.xlabel("Scor Z1 (Productie)")
plt.ylabel("Scor U1 (Emisii)")
plt.title("Instante in spatiul primei perechi canonice")
plt.grid()
plt.show() #