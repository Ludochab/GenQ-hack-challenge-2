# maxcut_gw.py
# Goemans–Williamson (SDP + random hyperplane rounding)
# - Si cvxpy est disponible : solveur SDP (SCS/OSQP/ECOS) + arrondi
# - Sinon : repli "spectral cut" (rapide, sans garantie 0.878)

import numpy as np
import networkx as nx

# -------------------------------
# Utilitaires
# -------------------------------

#Determine the section of the node. Either 0 or 1
def maxcut_value_from_assignment(W, x):
    """
    Calcule la valeur de la coupe pour une assignation binaire x ∈ {-1,+1}^n.
    Valeur = somme des poids des arêtes (i,j) telles que x_i != x_j.
    Hypothèse: W est symétrique, diagonale nulle, w_ij >= 0.
    """
    # Indicateur de séparation: 1 si signes différents, 0 sinon
    diff = (np.outer(x, x) < 0).astype(float)
    # Chaque arête (i,j) comptée deux fois dans W symétrique => diviser par 2
    return 0.5 * np.sum(W * diff)


def cut_from_assignment(x):
    """Retourne l'ensemble S = {i | x_i = +1} et son complément."""
    S = np.where(x > 0)[0].tolist()
    V_minus_S = np.where(x <= 0)[0].tolist()
    return S, V_minus_S


# -------------------------------
# Rounding GW: random hyperplane
# -------------------------------
def gw_rounding(V, W, R=64, rng=None):
    """
    Entrée:
      - V ∈ R^{n×d}: lignes = vecteurs v_i issus de la factorisation X = V V^T
      - W: matrice de poids (n×n)
      - R: nombre d'essais de plans aléatoires
    Sortie:
      - best_value, best_assignment (x ∈ {-1,+1}^n), (S, V\S)
    """
    n = V.shape[0]
    rng = np.random.default_rng(None if rng is None else rng)

    best_val = -np.inf
    best_x = None

    for _ in range(R):
        # Tirer un hyperplan aléatoire: vecteur gaussien d de dimension d
        d = rng.normal(size=(V.shape[1],))
        proj = V @ d  # <v_i, d>
        x = np.where(proj >= 0.0, 1.0, -1.0)
        val = maxcut_value_from_assignment(W, x)
        if val > best_val:
            best_val = val
            best_x = x

    return best_val, best_x, cut_from_assignment(best_x)


# -------------------------------
# SDP Goemans–Williamson (cvxpy)
# -------------------------------
def gw_sdp(W, solver_preference=("SCS", "ECOS", "OSQP")):
    """
    Résout la relaxation SDP:
      max sum_{i<j} w_ij * (1 - X_ij)/2
      s.c. X ⪰ 0, diag(X)=1
    Retourne une factorisation X ≈ V V^T (V = eigenvecs * sqrt(eigenvals_+)).
    """
    try:
        import cvxpy as cp
    except Exception:
        return None  # cvxpy indisponible -> le code appelant fera un repli spectral

    n = W.shape[0]
    X = cp.Variable((n, n), PSD=True)

    # Objectif: somme w_ij * (1 - X_ij)/2 sur i<j
    # On peut l'écrire sur tout i,j et diviser par 2 si la diagonale est nulle
    obj = cp.sum(cp.multiply(W, (1 - X))) / 2.0

    constraints = [cp.diag(X) == 1]
    prob = cp.Problem(cp.Maximize(obj), constraints)

    # Choisir un solveur disponible
    used = None
    for s in solver_preference:
        try:
            prob.solve(solver=getattr(cp, s))
            used = s
            break
        except Exception:
            continue

    if used is None or X.value is None:
        # Dernier essai: solveur par défaut
        prob.solve()
        if X.value is None:
            return None

    # Factorisation semi-définie positive: X ≈ Q Λ Q^T, ne garder que λ>=0
    Xval = 0.5 * (X.value + X.value.T)  # symétriser numériquement
    eigvals, eigvecs = np.linalg.eigh(Xval)
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    # Retirer les directions quasi-nulles pour stabilité numérique
    mask = eigvals_clipped > 1e-10
    if not np.any(mask):
        # tout nul -> retourner identité
        return np.eye(n)

    Lsqrt = np.sqrt(eigvals_clipped[mask])
    V = eigvecs[:, mask] * Lsqrt  # n×r
    return V


# -------------------------------
# Repli: Spectral cut (rapide)
# -------------------------------
def spectral_cut(W, R=1):
    """
    Baseline rapide:
      - Laplacien L = D - W
      - vecteur propre associé à la 2e plus petite valeur propre (Fiedler)
      - coupe selon le signe (ou seuil à 0)
    R est ignoré (compat signature).
    """
    # Laplacien
    d = np.sum(W, axis=1)
    L = np.diag(d) - W

    # Valeurs/vecteurs propres
    # On prend le vecteur propre de la 2e plus petite valeur propre (Fiedler).
    eigvals, eigvecs = np.linalg.eigh(L)
    if len(eigvals) < 2:
        # graphe trivial
        x = np.ones(W.shape[0])
    else:
        fiedler = eigvecs[:, 1]
        x = np.where(fiedler >= 0.0, 1.0, -1.0)

    val = maxcut_value_from_assignment(W, x)
    return val, x, cut_from_assignment(x)


# -------------------------------
# API principale
# -------------------------------
def maxcut_goemans_williamson(W, R=128, rng=None, verbose=True):
    """
    Essaie GW (SDP via cvxpy), sinon repli spectral.
    Entrées:
      - W: matrice de poids (n×n), symétrique, diag nulle
      - R: nombre d'essais d'arrondi pour GW
    Sorties:
      - best_value, best_assignment (x ∈ {-1,+1}^n), (S, V\S), meta
    """
    n = W.shape[0]
    assert W.shape[1] == n, "W doit être carrée"
    assert np.allclose(W, W.T, atol=1e-9), "W doit être symétrique"
    W = W.copy()
    np.fill_diagonal(W, 0.0)

    # 1) Tenter SDP (cvxpy)
    V = gw_sdp(W)
    if V is not None:
        best_val, best_x, (S, VS) = gw_rounding(V, W, R=R, rng=rng)
        meta = {"method": "GW+SDP", "R": R}
        if verbose:
            print(f"[GW] Valeur coupe = {best_val:.6f} (R={R})")
        return best_val, best_x, (S, VS), meta

    # 2) Repli spectral
    if verbose:
        print("[INFO] cvxpy indisponible -> repli Spectral Cut.")
    best_val, best_x, (S, VS) = spectral_cut(W)
    meta = {"method": "SpectralCut", "R": 1}
    if verbose:
        print(f"[Spectral] Valeur coupe = {best_val:.6f}")
    return best_val, best_x, (S, VS), meta


# -------------------------------
# Démo rapide
# -------------------------------
if __name__ == "__main__":
    # Petit graphe pondéré (triangle + une corde)
    # 0--1 (2.0), 1--2 (1.0), 0--2 (3.0), 2--3 (1.5)
    W = np.array([
        [0.0, 2.0, 3.0, 0.0],
        [2.0, 0.0, 1.0, 0.0],
        [3.0, 1.0, 0.0, 1.5],
        [0.0, 0.0, 1.5, 0.0],
    ], dtype=float)

    mtlMatrix = np.load("mtlFile.npy")

    #val, x, (S, VS), meta = maxcut_goemans_williamson(W, R=128, rng=42)

    val, x, (S, VS), meta = maxcut_goemans_williamson(mtlMatrix, R=128, rng=42)

    print("Méthode:", meta["method"])
    print("Coupe (S | V\\S):", S, "|", VS)
    print("Valeur de la coupe:", val)