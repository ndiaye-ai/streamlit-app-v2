import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Mini EDA 📊", page_icon="📊", layout="wide")

# --------------------- SIDEBAR ---------------------
st.sidebar.title("⚙️ Paramètres")
uploaded = st.sidebar.file_uploader("Uploader un CSV", type=["csv"])
use_demo = st.sidebar.checkbox("Utiliser les données de démo", value=not uploaded)

@st.cache_data
def load_demo_data(n=800, seed=123):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "categorie": rng.choice(["A", "B", "C"], size=n),
        "region": rng.choice(["Nord", "Sud", "Est", "Ouest"], size=n),
        "valeur": rng.normal(100, 20, size=n).round(2),
        "quantite": rng.integers(1, 100, size=n)
    })
    return df

@st.cache_data
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # tentative auto: parser des dates si une colonne s'appelle 'date'/'Date'
    for c in df.columns:
        if "date" in str(c).lower():
            with pd.option_context("mode.chained_assignment", None):
                try:
                    df[c] = pd.to_datetime(df[c], errors="ignore")
                except Exception:
                    pass
    return df

if use_demo:
    df = load_demo_data()
else:
    if uploaded is not None:
        df = load_csv(uploaded)
    else:
        st.info("👉 Uploade un CSV ou coche **Utiliser les données de démo**.")
        st.stop()

st.title("📊 Mini app d'exploration de données")
st.caption("Upload/filtre, indicateurs, aperçu et graphiques — avec export CSV/XLSX.")

# --------------------- DÉTECTION TYPES ---------------------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
dt_cols  = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
cat_cols = [c for c in df.columns if c not in num_cols and c not in dt_cols]

with st.sidebar.expander("🔎 Filtres"):
    # Filtres catégoriels (multiselect)
    cat_filters = {}
    for c in cat_cols:
        vals = sorted(df[c].dropna().astype(str).unique().tolist())
        default_vals = vals if len(vals) <= 30 else vals[:30]
        cat_filters[c] = st.multiselect(c, options=vals, default=default_vals)

    # Filtre date (range)
    date_filters = {}
    for c in dt_cols:
        dmin, dmax = pd.to_datetime(df[c].min()), pd.to_datetime(df[c].max())
        r = st.date_input(f"Période {c}", (dmin.date(), dmax.date()))
        if isinstance(r, tuple) and len(r) == 2:
            date_filters[c] = (pd.to_datetime(r[0]), pd.to_datetime(r[1]))

    # Filtre numérique (une colonne à la fois)
    num_filter_col = None
    num_range = None
    if num_cols:
        num_filter_col = st.selectbox("Filtrer une colonne numérique :", options=num_cols)
        vmin, vmax = float(df[num_filter_col].min()), float(df[num_filter_col].max())
        num_range = st.slider(f"Plage pour {num_filter_col}", vmin, vmax, (vmin, vmax))

# Appliquer les filtres
mask = pd.Series(True, index=df.index)

for c, selected in cat_filters.items():
    if selected and len(selected) < df[c].nunique():
        mask &= df[c].astype(str).isin(selected)

for c, (dmin, dmax) in date_filters.items():
    mask &= df[c].between(dmin, dmax)

if num_filter_col and num_range:
    mask &= df[num_filter_col].between(*num_range)

fdf = df.loc[mask].copy()

# --------------------- KPIs ---------------------
st.subheader("📌 Indicateurs")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Lignes filtrées", f"{len(fdf):,}".replace(",", " "))
if num_cols:
    kcol = st.selectbox("Colonne pour KPIs", options=num_cols, index=0, key="kpi_col")
    k2.metric(f"Moyenne({kcol})", f"{fdf[kcol].mean():.2f}")
    k3.metric(f"Min({kcol})", f"{fdf[kcol].min():.2f}")
    k4.metric(f"Max({kcol})", f"{fdf[kcol].max():.2f}")
else:
    k2.metric("—", "—"); k3.metric("—", "—"); k4.metric("—", "—")

st.divider()

# --------------------- APERÇU ---------------------
st.subheader("🧾 Aperçu des données")
st.dataframe(fdf.head(200), width="stretch")

# --------------------- CHARTS ---------------------
st.subheader("📈 Graphiques")
tab1, tab2 = st.tabs(["Série temporelle / ligne", "Barres agrégées"])

with tab1:
    if num_cols:
        if dt_cols:
            dcol = st.selectbox("Colonne de date :", options=dt_cols)
            ycol = st.selectbox("Valeur à tracer :", options=num_cols, index=min(1, len(num_cols)-1))
            freq = st.selectbox("Fréquence", ["D", "W", "M"], index=2, help="D=jour, W=semaine, M=mois")
            # Map pour compatibilité pandas (M -> ME : fin de mois)
            freq_map = {"D": "D", "W": "W", "M": "ME"}
            ts = (
                fdf[[dcol, ycol]]
                .dropna()
                .set_index(dcol)
                .sort_index()
                .resample(freq_map[freq])
                .mean(numeric_only=True)
            )
            st.line_chart(ts, height=360, width="stretch")
        else:
            ycol = st.selectbox("Valeur à tracer :", options=num_cols)
            st.line_chart(fdf[ycol], height=360, width="stretch")
    else:
        st.info("Aucune colonne numérique disponible pour tracer une courbe.")

with tab2:
    if num_cols:
        group_candidates = cat_cols + dt_cols
        if group_candidates:
            gcol = st.selectbox("Grouper par :", options=group_candidates)
            vcol = st.selectbox("Valeur :", options=num_cols, index=min(1, len(num_cols)-1), key="bar_val")
            agg = st.selectbox("Agrégat :", ["mean", "sum", "median", "max", "min"])
            g = getattr(fdf.groupby(gcol, dropna=False)[vcol], agg)().reset_index()
            # Si groupement par date → option de regrouper par mois
            if gcol in dt_cols:
                by = st.selectbox("Regrouper la date par :", ["Aucune", "Mois"], index=1)
                if by == "Mois":
                    tmp = fdf[[gcol, vcol]].dropna()
                    tmp["__mois__"] = tmp[gcol].dt.to_period("M").dt.to_timestamp()
                    g = getattr(tmp.groupby("__mois__")[vcol], agg)().reset_index().rename(columns={"__mois__": gcol})
            g = g.sort_values(vcol, ascending=False)
            st.bar_chart(g.set_index(g.columns[0])[vcol], height=360, width="stretch")
            with st.expander("Voir la table agrégée"):
                st.dataframe(g, width="stretch")
        else:
            st.info("Besoin d'au moins une colonne catégorielle ou date pour agréger.")
    else:
        st.info("Aucune colonne numérique à agréger.")

st.divider()

# --------------------- EXPORT ---------------------
st.subheader("📤 Export des données filtrées")
c1, c2 = st.columns(2)

csv = fdf.to_csv(index=False).encode("utf-8")
c1.download_button("⬇️ Télécharger en CSV", data=csv, file_name="donnees_filtrees.csv", mime="text/csv")

# Export XLSX en mémoire
output = BytesIO()
with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    fdf.to_excel(writer, index=False, sheet_name="Data")
output.seek(0)
c2.download_button("⬇️ Télécharger en Excel",
                   data=output.getvalue(),
                   file_name="donnees_filtrees.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with st.expander("ℹ️ Notes"):
    st.markdown(
        "- **Cache** via `@st.cache_data` pour accélérer le chargement.\n"
        "- Les colonnes dont le nom contient *date* sont automatiquement interprétées comme dates si possible.\n"
        "- Export disponible en **CSV** et **Excel** (XLSX)."
    )
