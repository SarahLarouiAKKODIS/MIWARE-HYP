import numpy as np
import pandas as pd


def display_class_distribution(y_train, y_test, save_path=None):
    """
    Affiche la distribution des classes dans les ensembles train et test.

    Parameters
    ----------
    y_train : np.ndarray
        Labels des pixels d'entraînement.
    y_test : np.ndarray
        Labels des pixels de test.
    save_path : str ou Path, optional
        Chemin du CSV à enregistrer.

    Returns
    -------
    pd.DataFrame
        Tableau de distribution des classes.
    """

    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()

    classes = np.union1d(np.unique(y_train), np.unique(y_test))

    train_counts = pd.Series(y_train).value_counts()
    test_counts = pd.Series(y_test).value_counts()

    df = pd.DataFrame({
        "Classe": classes,
        "Train (pixels)": [train_counts.get(c, 0) for c in classes],
        "Test (pixels)": [test_counts.get(c, 0) for c in classes],
    })

    df["Train (%)"] = (
        100 * df["Train (pixels)"] / len(y_train)
    ).round(2)

    df["Test (%)"] = (
        100 * df["Test (pixels)"] / len(y_test)
    ).round(2)

    df = df[
        ["Classe", "Train (pixels)", "Train (%)",
         "Test (pixels)", "Test (%)"]
    ]

    # Ligne TOTAL
    df.loc[len(df)] = [
        "TOTAL",
        len(y_train),
        100.0,
        len(y_test),
        100.0,
    ]

    print("\nDistribution des classes (Train / Test) :")
    print(df.to_string(index=False))

    if save_path is not None:
        df.to_csv(save_path, index=False)
        print(f"\nTableau sauvegardé : {save_path}")

    return df