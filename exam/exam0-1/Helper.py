import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hic  # For HCA
import sklearn.decomposition as dec  # For PCA
import sklearn.cross_decomposition as sk  # For CCA


# =============================================================================
# GLOBAL HELPER FUNCTIONS
# =============================================================================

# STANDARDISATION (Used in ALL exercises: PCA, HCA, CCA)
def stdd(A):
    # axis=0 means calculate along the columns
    means = np.mean(A, axis=0)
    std = np.std(A, axis=0)
    return (A - means) / std


# HCA THRESHOLD FINDER (The "Hard" Function)
# Finds the biggest jump in distance to cut the tree
def get_thr(h):
    dist = h[:, 2]  # Get distances
    m = len(dist)
    diff = dist[1:] - dist[:m - 1]  # Calculate differences
    diff_max = np.argmax(diff)  # Find index of max difference
    # Return average of the two steps at the jump
    return (h[diff_max, 2] + h[diff_max + 1, 2]) / 2


# =============================================================================
# HINTS: TASK 1 & 2 (DATA PREP)
# =============================================================================
"""
HINT 1: CALCULATING RATES
If asked for "Rate of X per 1000 inhabitants":
df["Rate_X"] = df["X"] * 1000 / df["Population"]

HINT 2: MERGING TABLES
If you have Demographics (nat) and Population (pop):
t_merged = nat.merge(pop, left_index=True, right_index=True)

HINT 3: REPLACING CODES WITH NAMES (MAPPING)
If you have 'CountyCode' and want 'CountyName':
1. Create a dictionary or use an existing column:
   map_dict = df.set_index('Code')['Name'].to_dict()
2. Apply it:
   df['CountyName'] = df['CountyCode'].map(map_dict)

HINT 4: INSERTING A COLUMN AT SPECIFIC POSITION
df.insert(1, "NewColName", values_array)
"""


# =============================================================================
# ALGORITHM 1: PCA (Principal Component Analysis)
# Goal: Reduce dimensions, find main trends.
# =============================================================================

def solve_pca(file_path):
    # 1. LOAD
    table = pd.read_csv(file_path, index_col=0)
    obs = table.index.values
    col = table.columns.values
    X = table.values

    # 2. STANDARDIZE
    x_std = stdd(X)

    # 3. MODEL
    objPCA = dec.PCA()
    objPCA.fit(x_std)

    # 4. SCORES (Principal Components) -> Task: "Save Principal Components"
    comp = objPCA.transform(x_std)
    # List comprehension for C1, C2, C3...
    comp_df = pd.DataFrame(comp, index=obs, columns=[f"C{i + 1}" for i in range(len(col))])
    comp_df.to_csv("./dataOUT/PrinComp.csv")

    # 5. EIGENVALUES & SCREE PLOT -> Task: "Plot Variance/Eigenvalues"
    eigenvalues = objPCA.explained_variance_
    plt.figure(figsize=(10, 7))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, "o-")
    plt.axhline(1, c='r', linestyle="--")  # Kaiser Criterion
    plt.title("Scree Plot")
    plt.show()

    # 6. LOADINGS & CIRCLE -> Task: "Correlation Circle"
    # Formula: Eigenvectors * sqrt(Eigenvalues)
    loadings = objPCA.components_.T * np.sqrt(objPCA.explained_variance_)

    plt.figure(figsize=(7, 7))
    plt.title("Correlation Circle")
    plt.gca().add_artist(plt.Circle((0, 0), 1, color='b', fill=False))

    for i in range(len(col)):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=0.03, color='k')
        plt.text(loadings[i, 0] + 0.05, loadings[i, 1] + 0.05, col[i], color='r')

    plt.xlim(-1.1, 1.1);
    plt.ylim(-1.1, 1.1)
    plt.show()


# =============================================================================
# ALGORITHM 2: HCA (Hierarchical Clustering)
# Goal: Group observations into clusters.
# =============================================================================

def solve_hca(file_path):
    # 1. LOAD & PREP
    table = pd.read_csv(file_path, index_col=0)
    obs = table.index.values
    X = table.values
    x_std = stdd(X)

    # 2. LINKAGE (Build Tree)
    linkage_matrix = hic.linkage(x_std, method="ward")

    # 3. FIND THRESHOLD (Uses helper function above)
    thr_val = get_thr(linkage_matrix)
    print("Threshold:", thr_val)

    # 4. PLOT DENDROGRAM
    plt.figure(figsize=(12, 7))
    plt.title("Dendrogram")
    hic.dendrogram(linkage_matrix, labels=obs, leaf_rotation=45, color_threshold=thr_val)
    plt.axhline(thr_val, c='r', linestyle="--")
    plt.show()

    # 5. PARTITION (Determine Clusters) -> Task: "Save Composition"
    cluster_labels = hic.fcluster(linkage_matrix, t=thr_val, criterion='distance')
    pd.DataFrame(cluster_labels, index=obs, columns=['Cluster']).to_csv("dataOUT/Xstd.csv")


# =============================================================================
# ALGORITHM 3: CCA (Canonical Correlation Analysis)
# Goal: Relate two sets of variables (X and Y).
# =============================================================================

def solve_cca(file_path):
    # 1. LOAD & SPLIT X/Y
    table = pd.read_csv(file_path, index_col=0)
    # EXAMPLE: First 4 cols are X, rest are Y
    x_col = table.columns[0:4]
    y_col = table.columns[4:]
    obs =table.index.values

    X = table[x_col].values
    Y = table[y_col].values

    # 2. STANDARDIZE
    x_std = stdd(X)
    y_std = stdd(Y)

    # 3. MODEL
    n = min(x_std.shape[1], y_std.shape[1])  # Min number of columns
    objCCA = sk.CCA(n_components=n)
    objCCA.fit(x_std, y_std)

    # 4. SCORES (z, u) -> Task: "Save Scores"
    z, u = objCCA.transform(x_std, y_std)
    # Save z (X scores) and u (Y scores)...

    # 5. LOADINGS (Rx, Ry) -> Task: "Factor Loadings"
    rx = objCCA.x_loadings_
    ry = objCCA.y_loadings_

    # 6. BIPLOT (Graphic) -> Task: "Distribution in space of roots"
    # Plot z1 vs z2 (X space)
    plt.figure(figsize=(15, 10))
    plt.title(label="Biplot")
    plt.xlabel(xlabel="z1,u1")
    plt.ylabel(ylabel="z2,u2")
    plt.scatter(x=z[:, 0], y=z[:, 1], c='r', label='Set X')
    plt.scatter(x=u[:, 0], y=u[:, 1], c='b', label='Set Y')
    for i in range(len()):
        plt.text(x=z[i, 0], y=z[i, 1], s=obs[i])
        plt.text(x=u[i, 0], y=u[i, 1], s=obs[i])
    plt.legend()
    plt.show()

