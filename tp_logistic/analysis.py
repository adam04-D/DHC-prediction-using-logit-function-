"""
TP : Régression logistique sous Python (équivalent du `glm` de R)
Cas d'étude : Prédiction de la maladie cardio-vasculaire (Cleveland Heart Disease)

Pipeline conforme au cours de Mme F. BADAOUI (INSEA) :
1.  Analyse descriptive
2.  Échantillonnage stratifié 80/20
3.  Ajustement d'un GLM Binomial(link=logit) par maximum de vraisemblance
4.  Tests d'hypothèses : Wald, rapport de vraisemblance, Pearson
5.  Adéquation : déviance, Hosmer-Lemeshow, pseudo-R², AIC, BIC
6.  Résidus : Pearson, déviance, leviers h_ii, distances de Cook
7.  Comparaison de modèles : matrice de confusion, courbe ROC
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit, probit, cloglog
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data", "heart_cleveland.csv")
RESULTS_TXT = os.path.join(ROOT, "results.txt")
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------------------
# 0.  Capture stdout dans un fichier "results.txt" pour le rapport
# ---------------------------------------------------------------------------
import io, sys
_buffer = io.StringIO()


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, msg):
        for s in self.streams:
            s.write(msg)

    def flush(self):
        for s in self.streams:
            s.flush()


sys.stdout = _Tee(sys.__stdout__, _buffer)


def H(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1.  Chargement et préparation des données
# ---------------------------------------------------------------------------
H("1. CHARGEMENT ET PRÉPARATION DES DONNÉES")

cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

df = pd.read_csv(DATA, header=None, names=cols, na_values="?")
print(f"Dimensions brutes : {df.shape}")
print(f"Valeurs manquantes par variable :\n{df.isna().sum()}")

# Suppression des lignes incomplètes (ca, thal)
df = df.dropna().reset_index(drop=True)
print(f"\nDimensions après suppression des NA : {df.shape}")

# Variable réponse binaire : 0 = absence, 1 = présence d'une maladie
df["chd"] = (df["num"] > 0).astype(int)
df = df.drop(columns=["num"])

# Recodage des variables qualitatives (cf. cours, p. 153 et suiv.)
df["sex"] = df["sex"].map({0: "F", 1: "M"})
df["cp"] = df["cp"].map({1: "angine_typ", 2: "angine_atyp",
                          3: "non_angineuse", 4: "asymptomatique"})
df["fbs"] = df["fbs"].map({0: "non", 1: "oui"})
df["restecg"] = df["restecg"].map({0: "normal", 1: "anomalie_ST_T",
                                    2: "hypertrophie_VG"})
df["exang"] = df["exang"].map({0: "non", 1: "oui"})
df["slope"] = df["slope"].map({1: "ascendante", 2: "plate", 3: "descendante"})
df["thal"] = df["thal"].map({3.0: "normal", 6.0: "fixe", 7.0: "reversible"})

print("\nAperçu des données :")
print(df.head())
print("\nDescriptif des variables quantitatives :")
print(df.describe().round(2))

# ---------------------------------------------------------------------------
# 2.  Analyse descriptive
# ---------------------------------------------------------------------------
H("2. ANALYSE DESCRIPTIVE DE LA VARIABLE RÉPONSE")

print(df["chd"].value_counts())
print(f"\nProportion de malades : {df['chd'].mean():.3f}")
print(f"Proportion de sains   : {1 - df['chd'].mean():.3f}")
print("→ La variable réponse n'est pas parfaitement équilibrée mais reste exploitable.")
print("→ Un échantillonnage stratifié est utilisé pour préserver la répartition.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.countplot(x="chd", data=df, ax=axes[0], palette="Set2")
axes[0].set_title("Distribution de la variable réponse CHD")
axes[0].set_xlabel("CHD (0=sain, 1=malade)")
axes[0].set_ylabel("Effectif")

sns.boxplot(x="chd", y="age", data=df, ax=axes[1], palette="Set2")
axes[1].set_title("Âge selon CHD")
axes[1].set_xlabel("CHD")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "01_distribution_reponse.png"))
plt.close()

# Liaison variables continues / réponse
quant = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, c in zip(axes.flatten(), quant):
    sns.boxplot(x="chd", y=c, data=df, ax=ax, palette="Set2")
    ax.set_title(c)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "02_quantitatives_vs_chd.png"))
plt.close()

# Liaison variables qualitatives / réponse
qual = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for ax, c in zip(axes.flatten(), qual):
    pd.crosstab(df[c], df["chd"], normalize="index").plot(
        kind="bar", stacked=True, ax=ax, colormap="Set2", legend=False)
    ax.set_title(c); ax.set_xlabel("")
axes.flatten()[-1].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "03_qualitatives_vs_chd.png"))
plt.close()

# Matrice de corrélation
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(df[quant + ["chd"]].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", ax=ax)
ax.set_title("Matrice de corrélation (variables quantitatives)")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "04_correlation.png"))
plt.close()

# ---------------------------------------------------------------------------
# 3.  Préparation du design + échantillonnage stratifié 80/20
# ---------------------------------------------------------------------------
H("3. ÉCHANTILLONNAGE STRATIFIÉ 80/20")

# Encodage one-hot avec modalité de référence (drop_first)
X_full = pd.get_dummies(
    df.drop(columns="chd"),
    columns=qual, drop_first=True
).astype(float)
y_full = df["chd"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.20, stratify=y_full, random_state=42
)
print(f"Apprentissage : {X_train.shape}, prop CHD = {y_train.mean():.3f}")
print(f"Validation    : {X_test.shape}, prop CHD = {y_test.mean():.3f}")

# Standardisation des continues (utile pour la stabilité numérique
# et la comparabilité des coefficients)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train[quant] = scaler.fit_transform(X_train[quant])
X_test[quant] = scaler.transform(X_test[quant])

X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)
X_test_sm = X_test_sm[X_train_sm.columns]  # même ordre

# ---------------------------------------------------------------------------
# 4.  Ajustement du GLM logistique complet (M_full)
# ---------------------------------------------------------------------------
H("4. AJUSTEMENT DU GLM BINOMIAL (LIEN LOGIT) – MODÈLE COMPLET")

m_full = sm.GLM(y_train, X_train_sm, family=Binomial(link=logit())).fit()
print(m_full.summary())

# ---------------------------------------------------------------------------
# 5.  Sélection backward fondée sur l'AIC (cf. cours p. 117 et 217)
# ---------------------------------------------------------------------------
H("5. SÉLECTION DE VARIABLES PAR BACKWARD AIC")


def backward_aic(X, y, verbose=False):
    cur = list(X.columns)
    best_aic = sm.GLM(y, X[cur], family=Binomial(link=logit())).fit().aic
    improved = True
    while improved and len(cur) > 1:
        improved = False
        scores = []
        for c in cur:
            if c == "const":
                continue
            trial = [k for k in cur if k != c]
            try:
                a = sm.GLM(y, X[trial], family=Binomial(link=logit())).fit().aic
                scores.append((a, c))
            except Exception:
                continue
        scores.sort()
        if scores and scores[0][0] < best_aic - 1e-3:
            best_aic = scores[0][0]
            cur.remove(scores[0][1])
            if verbose:
                print(f"  - retire {scores[0][1]:<25s}  AIC={best_aic:.3f}")
            improved = True
    return cur


sel = backward_aic(X_train_sm, y_train, verbose=True)
print(f"\nVariables retenues ({len(sel)-1}) : {[c for c in sel if c!='const']}")

m_red = sm.GLM(y_train, X_train_sm[sel], family=Binomial(link=logit())).fit()
print("\n-- Modèle réduit --")
print(m_red.summary())

# Modèle "naïf" (uniquement age) pour comparaison
m_age = sm.GLM(y_train, X_train_sm[["const", "age"]],
               family=Binomial(link=logit())).fit()
print("\n-- Modèle simple (age uniquement) --")
print(m_age.summary())

# ---------------------------------------------------------------------------
# 6.  Comparaison des modèles emboîtés (test du rapport de vraisemblance)
# ---------------------------------------------------------------------------
H("6. TEST DU RAPPORT DE VRAISEMBLANCE (modèles emboîtés)")


def lr_test(small, big, name1, name2):
    stat = small.deviance - big.deviance
    ddl = int(small.df_resid - big.df_resid)
    p = 1 - stats.chi2.cdf(stat, ddl)
    print(f"H0: {name1}  vs  H1: {name2}")
    print(f"  ΔD = D({name1}) - D({name2}) = {stat:.4f}")
    print(f"  ddl = {ddl},  p-value = {p:.5g}")
    decision = "H0 rejetée" if p < 0.05 else "H0 conservée"
    print(f"  → {decision} au seuil 5%\n")


lr_test(m_age, m_red, "M_age", "M_reduit")
lr_test(m_red, m_full, "M_reduit", "M_complet")

# ---------------------------------------------------------------------------
# 7.  Adéquation du modèle : déviance, Pearson, Hosmer-Lemeshow, pseudo-R²
# ---------------------------------------------------------------------------
H("7. ADÉQUATION DU MODÈLE RETENU (M_reduit)")


def adequation(model, y, name):
    n = len(y)
    p = model.df_model
    D = model.deviance
    ddl = model.df_resid
    pchi2 = float(((y - model.fittedvalues) ** 2 /
                   (model.fittedvalues * (1 - model.fittedvalues))).sum())
    pseudoR2 = 1 - model.deviance / model.null_deviance
    print(f"--- {name} ---")
    print(f"  n = {n}, p (sans intercept) = {int(p)}, ddl résiduels = {int(ddl)}")
    print(f"  Déviance D       = {D:.4f}    (ratio D/ddl = {D/ddl:.3f})")
    print(f"  χ² Pearson       = {pchi2:.4f}  (ratio = {pchi2/ddl:.3f})")
    print(f"  Déviance nulle D0 = {model.null_deviance:.4f}")
    print(f"  Pseudo-R² (McFadden via déviance) = {pseudoR2:.4f}")
    print(f"  AIC = {model.aic:.3f}   BIC = {model.bic_llf:.3f}")
    p_dev = 1 - stats.chi2.cdf(D, ddl)
    print(f"  Test de déviance : p-value = {p_dev:.4g} "
          f"({'modèle adéquat' if p_dev > 0.05 else 'inadéquat'})")


adequation(m_age, y_train, "M_age")
adequation(m_red, y_train, "M_reduit")
adequation(m_full, y_train, "M_complet")


def hosmer_lemeshow(y_true, y_prob, g=10):
    """Test de Hosmer-Lemeshow (cf. cours p. 165-166)."""
    df_ = pd.DataFrame({"y": y_true, "p": y_prob})
    df_["bin"] = pd.qcut(df_["p"], q=g, duplicates="drop")
    obs = df_.groupby("bin")["y"].agg(["sum", "count"])
    exp = df_.groupby("bin")["p"].agg(["sum", "mean"])
    o1 = obs["sum"].values
    n_ = obs["count"].values
    pi_ = exp["mean"].values
    e1 = n_ * pi_
    C = np.sum((o1 - e1) ** 2 / (n_ * pi_ * (1 - pi_)))
    df_test = len(o1) - 2
    p_val = 1 - stats.chi2.cdf(C, df_test)
    return C, df_test, p_val


C, df_hl, p_hl = hosmer_lemeshow(y_train, m_red.fittedvalues, g=10)
print(f"\nHosmer-Lemeshow (modèle réduit, g=10) : C² = {C:.4f}, "
      f"ddl = {df_hl}, p = {p_hl:.4g}")
print("→ p > 0.05 indique un bon ajustement.")

# ---------------------------------------------------------------------------
# 8.  Analyse des résidus (Pearson, déviance, leviers, Cook)
# ---------------------------------------------------------------------------
H("8. ANALYSE DES RÉSIDUS (modèle retenu)")

infl = m_red.get_influence()
res_p = m_red.resid_pearson
res_d = m_red.resid_deviance
lev = infl.hat_matrix_diag
cook = infl.cooks_distance[0]

print(f"|résidus de Pearson|  > 2 : {(np.abs(res_p) > 2).sum()} obs")
print(f"|résidus de déviance| > 2 : {(np.abs(res_d) > 2).sum()} obs")
seuil_h = 2 * (m_red.df_model + 1) / len(y_train)
print(f"Seuil de levier h_ii > 2(p+1)/n = {seuil_h:.4f} : "
      f"{(lev > seuil_h).sum()} obs influentes")
print(f"Distance de Cook max = {cook.max():.4f}  (alerte si > 1)")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes[0, 0].scatter(m_red.fittedvalues, res_p, alpha=0.6)
axes[0, 0].axhline(0, c="red"); axes[0, 0].axhline(2, c="red", ls="--")
axes[0, 0].axhline(-2, c="red", ls="--")
axes[0, 0].set_xlabel("Valeurs ajustées π̂")
axes[0, 0].set_ylabel("Résidus de Pearson")
axes[0, 0].set_title("Résidus de Pearson")

axes[0, 1].scatter(m_red.fittedvalues, res_d, alpha=0.6, c="darkorange")
axes[0, 1].axhline(0, c="red"); axes[0, 1].axhline(2, c="red", ls="--")
axes[0, 1].axhline(-2, c="red", ls="--")
axes[0, 1].set_xlabel("Valeurs ajustées π̂")
axes[0, 1].set_ylabel("Résidus de déviance")
axes[0, 1].set_title("Résidus de déviance")

stats.probplot(res_d, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title("QQ-plot des résidus de déviance")

axes[1, 1].stem(np.arange(len(cook)), cook, markerfmt=",")
axes[1, 1].axhline(4 / len(y_train), c="red", ls="--",
                   label=f"seuil 4/n = {4/len(y_train):.3f}")
axes[1, 1].set_xlabel("Indice de l'observation")
axes[1, 1].set_ylabel("Distance de Cook")
axes[1, 1].set_title("Distances de Cook")
axes[1, 1].legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "05_residus.png"))
plt.close()

# ---------------------------------------------------------------------------
# 9.  Comparaison sur l'échantillon test : matrice de confusion + ROC
# ---------------------------------------------------------------------------
H("9. ÉVALUATION SUR L'ÉCHANTILLON DE VALIDATION (n=20%)")


def evaluate(model, X_te, y_te, name, seuil=0.5):
    p_hat = model.predict(X_te)
    y_hat = (p_hat >= seuil).astype(int)
    cm = confusion_matrix(y_te, y_hat)
    acc = accuracy_score(y_te, y_hat)
    prec = precision_score(y_te, y_hat, zero_division=0)
    rec = recall_score(y_te, y_hat)
    f1 = f1_score(y_te, y_hat)
    fpr, tpr, _ = roc_curve(y_te, p_hat)
    a = auc(fpr, tpr)
    print(f"\n--- {name} (seuil={seuil}) ---")
    print(f"Matrice de confusion :\n{cm}")
    print(f"  Accuracy={acc:.3f}  Précision={prec:.3f}  "
          f"Rappel={rec:.3f}  F1={f1:.3f}  AUC={a:.3f}")
    return {"name": name, "cm": cm, "fpr": fpr, "tpr": tpr,
            "auc": a, "acc": acc, "f1": f1}


eval_full = evaluate(m_full, X_test_sm, y_test, "M_complet")
eval_red = evaluate(m_red, X_test_sm[sel], y_test, "M_reduit")
eval_age = evaluate(m_age, X_test_sm[["const", "age"]], y_test, "M_age")

# Courbe ROC comparée
fig, ax = plt.subplots(figsize=(7, 6))
for r in [eval_age, eval_red, eval_full]:
    ax.plot(r["fpr"], r["tpr"], label=f"{r['name']} (AUC={r['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax.set_xlabel("Taux de faux positifs")
ax.set_ylabel("Taux de vrais positifs (sensibilité)")
ax.set_title("Courbes ROC – comparaison des modèles")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "06_roc.png"))
plt.close()

# Matrices de confusion graphiques
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, r in zip(axes, [eval_age, eval_red, eval_full]):
    sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["Réel 0", "Réel 1"], cbar=False)
    ax.set_title(f"{r['name']}\nAcc={r['acc']:.3f}  AUC={r['auc']:.3f}")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "07_confusion_matrices.png"))
plt.close()

# ---------------------------------------------------------------------------
# 10. Interprétation : odds ratios du modèle retenu
# ---------------------------------------------------------------------------
H("10. INTERPRÉTATION DU MODÈLE RETENU – ODDS RATIOS")

params = m_red.params
ci = m_red.conf_int()
or_table = pd.DataFrame({
    "Coefficient β": params,
    "OR = exp(β)": np.exp(params),
    "IC95% bas": np.exp(ci[0]),
    "IC95% haut": np.exp(ci[1]),
    "p-value": m_red.pvalues
}).round(4)
print(or_table.to_string())

# Sauvegarde du tableau
or_table.to_csv(os.path.join(ROOT, "odds_ratios.csv"))

# ---------------------------------------------------------------------------
# 11. Comparaison de fonctions de lien (logit vs probit vs cloglog)
# ---------------------------------------------------------------------------
H("11. COMPARAISON DE FONCTIONS DE LIEN (logit / probit / cloglog)")

links = {"logit": logit(), "probit": probit(), "cloglog": cloglog()}
for nm, lk in links.items():
    mdl = sm.GLM(y_train, X_train_sm[sel], family=Binomial(link=lk)).fit()
    print(f"  {nm:8s}  Déviance = {mdl.deviance:.3f}   AIC = {mdl.aic:.3f}")

# ---------------------------------------------------------------------------
# Sauvegarde des résultats texte
# ---------------------------------------------------------------------------
with open(RESULTS_TXT, "w") as f:
    f.write(_buffer.getvalue())

print("\n[OK] Tous les résultats texte ont été enregistrés dans results.txt")
print(f"[OK] Figures enregistrées dans {FIG}")
