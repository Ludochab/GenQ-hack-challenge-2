import numpy as np

# -------------------------------
# Utilitaires Max-Cut
# -------------------------------
def cut_value(W, x):
    """
    Valeur de la coupe pour x ∈ {-1,+1}^n :
    sum_{i<j} w_ij [x_i != x_j].
    Implémentation via matrice (en comptant chaque arête deux fois et en divisant par 2).
    """
    diff = (np.outer(x, x) < 0).astype(float)
    return 0.5 * np.sum(W * diff)

def cut_sets_from_x(x):
    S = np.where(x > 0)[0].tolist()
    T = np.where(x <= 0)[0].tolist()
    return S, T

# -------------------------------
# 1) Seed à partir de I1, I2
# -------------------------------
def seed_from_independent_sets(W, I1, I2):
    """
    Crée une assignation x initiale :
      x_i=+1 si i∈I1 ; x_i=-1 si i∈I2 ; non assignés sinon (0).
    Retourne x (float), et le masque des sommets déjà assignés.
    """
    n = W.shape[0]
    x = np.zeros(n, dtype=float)
    assigned = np.zeros(n, dtype=bool)
    for i in I1:
        x[i] = 1.0
        assigned[i] = True
    for i in I2:
        x[i] = -1.0
        assigned[i] = True
    return x, assigned

# -------------------------------
# 2) Affectation gloutonne des sommets restants
# -------------------------------
def assign_remaining_vertices(W, x, assigned):
    """
    Pour chaque sommet non assigné i, on place i du côté qui maximise
    la contribution de coupe *par rapport aux voisins déjà assignés*.
    Formule : choisir +1 si s = sum_{j assignés} w_ij x_j <= 0, sinon -1.
    """
    n = W.shape[0]
    # ordre : degrés pondérés décroissants (optionnel, souvent un peu mieux)
    deg_w = W.sum(axis=1)
    order = np.argsort(-deg_w)
    for i in order:
        if assigned[i]:
            continue
        # somme partielle s = ∑_{j assignés} w_ij x_j
        s = np.dot(W[i, assigned], x[assigned])
        x[i] = 1.0 if s <= 0 else -1.0
        assigned[i] = True
    return x

# -------------------------------
# 3) Amélioration locale 1-opt efficace
# -------------------------------
def improve_1opt(W, x, max_passes=10):
    """
    Amélioration locale : on flippe un sommet i si cela augmente la coupe.
    Identité clé : le gain Δ de flipper i vaut Δ_i = x_i * (W @ x)_i.
    On maintient y = W @ x et on le met à jour en O(deg) à chaque flip.
    """
    n = W.shape[0]
    x = x.copy()
    y = W @ x  # y_i = ∑_j w_ij x_j
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        # Itérer dans un ordre aléatoire (souvent mieux qu'un ordre fixe)
        for i in np.random.permutation(n):
            delta = x[i] * y[i]   # gain si on flippe i
            if delta > 1e-12:
                # flip: x_i ← -x_i
                old_xi = x[i]
                x[i] = -x[i]
                # mise à jour de y = W @ x :
                # x' = x - 2*old_xi*e_i  => y' = y - 2*old_xi*W[:,i]
                y = y - 2.0 * old_xi * W[:, i]
                improved = True
    return x

# -------------------------------
# 4) (Option) Goemans–Williamson biaisé par I1/I2
# -------------------------------
def gw_sdp_factor(W, solver_preference=("SCS","ECOS","OSQP")):
    """
    Solve la relaxation SDP :
      max sum w_ij (1 - X_ij)/2  s.c. X ⪰ 0, diag(X)=1
    Retourne V tel que X ≈ V V^T (ou None si cvxpy non dispo / échec).
    """
    try:
        import cvxpy as cp
    except Exception:
        return None

    n = W.shape[0]
    X = cp.Variable((n, n), PSD=True)
    obj = cp.sum(cp.multiply(W, (1 - X))) / 2.0
    prob = cp.Problem(cp.Maximize(obj), [cp.diag(X) == 1])

    used = None
    for s in solver_preference:
        try:
            prob.solve(solver=getattr(cp, s))
            used = s
            break
        except Exception:
            continue
    if used is None or X.value is None:
        prob.solve()
        if X.value is None:
            return None

    Xval = 0.5 * (X.value + X.value.T)
    eigvals, eigvecs = np.linalg.eigh(Xval)
    eigvals = np.clip(eigvals, 0.0, None)
    mask = eigvals > 1e-10
    if not np.any(mask):
        return np.eye(n)
    V = eigvecs[:, mask] * np.sqrt(eigvals[mask])
    return V

def gw_rounding_biased(V, W, I1=None, I2=None, mu_list=(0.0, 0.2, 0.4), R=64, rng=None):
    """
    Arrondi GW avec biais : x_i = sign(<v_i, r> + μ b_i),
      b_i = +1 (i∈I1), -1 (i∈I2), 0 sinon.
    On essaie plusieurs μ et plusieurs tirages r, on garde la meilleure coupe.
    """
    rng = np.random.default_rng(rng)
    n, d = V.shape
    b = np.zeros(n, dtype=float)
    if I1 is not None:
        b[list(I1)] =  1.0
    if I2 is not None:
        b[list(I2)] = -1.0

    best_val, best_x = -np.inf, None
    for mu in mu_list:
        for _ in range(R):
            r = rng.normal(size=d)
            proj = V @ r + mu * b
            x = np.where(proj >= 0.0, 1.0, -1.0)
            val = cut_value(W, x)
            if val > best_val:
                best_val, best_x = val, x.copy()
    return best_val, best_x

# -------------------------------
# Pipeline principal
# -------------------------------
def maxcut_hybrid_with_seeds(W, I1, I2, use_gw=True, R=128, mu_list=(0.0,0.2,0.4), rng=42, verbose=True):
    """
    1) Seed via I1/I2 + affectation des autres sommets
    2) 1-opt
    3) (Option) GW biaisé par I1/I2, puis 1-opt final
    Retourne la meilleure solution et des métadonnées.
    """
    W = np.array(W, dtype=float)
    n = W.shape[0]
    assert W.shape[1] == n, "W doit être carrée"
    assert np.allclose(W, W.T, atol=1e-9), "W doit être symétrique"
    np.fill_diagonal(W, 0.0)

    # (1) Seed
    x, assigned = seed_from_independent_sets(W, I1, I2)
    x = assign_remaining_vertices(W, x, assigned)
    # (2) 1-opt
    x = improve_1opt(W, x)
    val_seed = cut_value(W, x)
    best_val, best_x, best_meta = val_seed, x.copy(), {"method": "Seed+1opt"}

    if verbose:
        print(f"[Seed+1opt] Valeur coupe = {val_seed:.6f}")

    # (3) Optionnel : GW biaisé + 1-opt
    if use_gw:
        V = gw_sdp_factor(W)
        if V is None:
            if verbose:
                print("[INFO] cvxpy indisponible ou SDP échoué -> on garde Seed+1opt.")
        else:
            gw_val, gw_x = gw_rounding_biased(V, W, I1=I1, I2=I2, mu_list=mu_list, R=R, rng=rng)
            # petit polissage 1-opt
            gw_x = improve_1opt(W, gw_x)
            gw_val = cut_value(W, gw_x)
            if verbose:
                print(f"[GW biaisé + 1opt] Valeur coupe = {gw_val:.6f}")
            if gw_val > best_val:
                best_val, best_x, best_meta = gw_val, gw_x.copy(), {"method": "GW_biased+1opt", "R": R, "mu_list": mu_list}

    S, T = cut_sets_from_x(best_x)
    return best_val, best_x, (S, T), best_meta

# -------------------------------
# Démo minimale
# -------------------------------
if __name__ == "__main__":
    # Exemple : petit graphe pondéré (4 sommets)
    # 0--1 (2.0), 1--2 (1.0), 0--2 (3.0), 2--3 (1.5)
    W = np.array([
        [0.0, 2.0, 3.0, 0.0],
        [2.0, 0.0, 1.0, 0.0],
        [3.0, 1.0, 0.0, 1.5],
        [0.0, 0.0, 1.5, 0.0],
    ], dtype=float)

    # Supposons qu'on a deux ensembles indépendants I1, I2 (indices de sommets)
    I1 = {0}   # {0} est indépendant ici
    I2 = {3}   # {3} aussi

    best_val, best_x, (S, T), meta = maxcut_hybrid_with_seeds(W, I1, I2, use_gw=True, R=64, mu_list=(0.0,0.3,0.6), rng=123, verbose=True)
    print("Méthode :", meta)
    print("Coupe (S | T):", S, "|", T)
    print("Valeur de la coupe :", best_val)