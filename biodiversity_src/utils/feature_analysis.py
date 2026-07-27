import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def plot_pca_by_class_and_group(
    X,
    y,
    groups,
    max_samples=50000,
    random_state=42,
):
    """
    Affiche une ACP des features :
    - colorée par classe d'espèce
    - colorée par site/groupe

    X      : (n_pixels, n_features)
    y      : labels espèces (n_pixels,)
    groups : identifiant site/image/groupe (n_pixels,)
    """

    rng = np.random.default_rng(random_state)

    n = X.shape[0]
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X_plot = X[idx]
        y_plot = y[idx]
        groups_plot = groups[idx]
    else:
        X_plot = X
        y_plot = y
        groups_plot = groups

    valid = np.isfinite(X_plot).all(axis=1)
    X_plot = X_plot[valid]
    y_plot = y_plot[valid]
    groups_plot = groups_plot[valid]

    print("Nombre d'échantillons utilisés :", X_plot.shape[0])

    X_scaled = StandardScaler().fit_transform(X_plot)

    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    print("Variance expliquée PC1/PC2 :", pca.explained_variance_ratio_)

    # ACP colorée par espèce
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y_plot,
        s=3,
        alpha=0.5,
        cmap="tab20",
    )
    plt.colorbar(scatter, label="Classe espèce")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("ACP des features - couleur = espèce")
    plt.tight_layout()
    plt.show()

    # ACP colorée par site/groupe
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=groups_plot,
        s=3,
        alpha=0.5,
        cmap="tab10",
    )
    plt.colorbar(scatter, label="Site / groupe")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("ACP des features - couleur = site/groupe")
    plt.tight_layout()
    plt.show()


    print(pca.explained_variance_ratio_)
