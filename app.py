"""
PharmaLens — Drug Intelligence Dashboard
==========================================
A Streamlit dashboard for pharmacists and consumers built around a CatBoost
"drug condition-category" classifier (drug_classifier_package.pkl).

Run with:  streamlit run app.py
See README.md for full setup instructions.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from lib.data_gen import generate_drug_catalog, validate_uploaded_catalog, REQUIRED_COLUMNS
from lib.model_loader import load_model_package
from lib.preprocessing import predict_condition_category
from lib import calculators as calc

# --------------------------------------------------------------------------------------
# Page config & light theming
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="PharmaLens — Drug Intelligence Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .metric-card {
        background: linear-gradient(135deg, #0f766e 0%, #134e4a 100%);
        padding: 1.1rem 1.3rem; border-radius: 14px; color: white;
    }
    .disclaimer-box {
        background-color: #fff7ed; border-left: 5px solid #ea580c;
        padding: 0.8rem 1rem; border-radius: 8px; font-size: 0.9rem;
    }
    .pill {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        background: #e0f2fe; color: #075985; font-size: 0.78rem; margin: 2px;
    }
    section[data-testid="stSidebar"] { background-color: #f8fafc; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

GLOBAL_DISCLAIMER = (
    "**⚠️ Educational tool, not medical advice.** PharmaLens uses a machine-learning model "
    "trained on synthetic/demo data and a demo drug catalog. It is intended to illustrate "
    "dashboard capabilities — always verify against official prescribing information (e.g. "
    "FDA label, pharmacist, or physician) before making any clinical or personal health decision."
)

# --------------------------------------------------------------------------------------
# Data / model loading
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Building demo drug catalog...")
def get_default_catalog():
    return generate_drug_catalog(n=420, seed=42)


def get_catalog() -> pd.DataFrame:
    if "user_catalog" in st.session_state and st.session_state["user_catalog"] is not None:
        return st.session_state["user_catalog"]
    return get_default_catalog()


@st.cache_data(show_spinner=False)
def build_side_effect_frequency(df: pd.DataFrame) -> pd.DataFrame:
    counts = {}
    for effects in df["side_effects"].dropna():
        for e in [x.strip().lower() for x in str(effects).split(",") if x.strip()]:
            counts[e] = counts.get(e, 0) + 1
    out = pd.DataFrame(sorted(counts.items(), key=lambda kv: kv[1], reverse=True),
                        columns=["side_effect", "count"])
    return out


@st.cache_resource(show_spinner=False)
def build_similarity_index(_df_key: tuple, df: pd.DataFrame):
    """TF-IDF + cosine similarity over side_effects/condition/warnings for the
    'Similar Drug Finder' feature. This is a lightweight, catalog-local index —
    independent from the packaged CatBoost model's own TF-IDF vectorizer."""
    text = (df["condition"].fillna("") + " " + df["side_effects"].fillna("") + " " +
            df["warnings"].fillna(""))
    vec = TfidfVectorizer(max_features=3000, stop_words="english")
    matrix = vec.fit_transform(text)
    return vec, matrix


model_package, model_error = load_model_package()

# --------------------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------------------
st.sidebar.markdown("## 💊 PharmaLens")
st.sidebar.caption("Drug Intelligence Dashboard")

mode = st.sidebar.radio("Viewing as", ["🧑‍⚕️ Pharmacist", "🙋 Consumer"], index=0)
st.session_state["mode"] = mode

PAGES = [
    "🏠 Home",
    "📊 Analytics",
    "🔍 Search & Lookup",
    "⚖️ Compare Drugs",
    "🤖 AI Category Classifier",
    "🧬 Similar Drug Finder",
    "🧪 Interaction Checker",
    "🧮 Calculators & Tools",
    "📅 Medication Tracker",
    "ℹ️ About This Model",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.divider()
with st.sidebar.expander("📂 Use your own drug catalog (CSV)"):
    st.caption(f"Required columns: {', '.join(REQUIRED_COLUMNS)}")
    upl = st.file_uploader("Upload CSV", type=["csv"], key="catalog_upload")
    if upl is not None:
        try:
            new_df = pd.read_csv(upl)
            ok, missing = validate_uploaded_catalog(new_df)
            if ok:
                st.session_state["user_catalog"] = new_df
                st.success(f"Loaded {len(new_df)} rows.")
            else:
                st.error(f"Missing columns: {missing}")
        except Exception as e:
            st.error(f"Could not read file: {e}")
    if st.session_state.get("user_catalog") is not None:
        if st.button("Reset to demo catalog"):
            st.session_state["user_catalog"] = None
            st.rerun()

if model_error:
    st.sidebar.error(f"Model not loaded: {model_error}")
else:
    st.sidebar.success("ML model package loaded ✓")

df = get_catalog()

# ========================================================================================
# PAGE: HOME
# ========================================================================================
if page == "🏠 Home":
    st.title("💊 PharmaLens — Drug Intelligence Dashboard")
    st.markdown(GLOBAL_DISCLAIMER, help=None)
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Drugs in catalog", f"{len(df):,}")
    c2.metric("Therapeutic categories", df["condition_category"].nunique())
    c3.metric("Avg. effectiveness score", f"{df['effectiveness_score'].mean():.1f} / 10")
    c4.metric("OTC share", f"{(df['otc_or_rx'] == 'OTC').mean()*100:.0f}%")

    st.write("")
    left, right = st.columns([2, 1])
    with left:
        st.subheader("What's inside")
        st.markdown("""
- **📊 Analytics** — top side effects, top/most-effective drugs, category & pricing breakdowns
- **🔍 Search & Lookup** — filterable drug search engine
- **⚖️ Compare Drugs** — side-by-side comparison of up to 4 drugs
- **🤖 AI Category Classifier** — runs the packaged CatBoost model on a drug label you describe
- **🧬 Similar Drug Finder** — content-based "find alternatives" tool
- **🧪 Interaction Checker** — quick keyword-based interaction lookup across selected drugs
- **🧮 Calculators & Tools** — BMI, dose-from-concentration, dilution (C1V1), pediatric weight dosing, creatinine clearance, unit converter
- **📅 Medication Tracker** — build a simple daily dosing schedule (session-only)
        """)
    with right:
        st.subheader("Category mix")
        cat_counts = df["condition_category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        fig = px.pie(cat_counts, names="category", values="count", hole=0.5)
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=320)
        fig.update_traces(textinfo="label", textfont_size=10)
        st.plotly_chart(fig, use_container_width=True)

# ========================================================================================
# PAGE: ANALYTICS
# ========================================================================================
elif page == "📊 Analytics":
    st.title("📊 Analytics Dashboard")
    st.caption("Aggregated insights across the active drug catalog.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Side Effects", "Top & Most Effective Drugs", "Category Breakdown", "Pricing & Safety"]
    )

    with tab1:
        st.subheader("Most frequently reported side effects")
        freq = build_side_effect_frequency(df)
        top_n = st.slider("Show top N", 5, 30, 15, key="se_topn")
        fig = px.bar(freq.head(top_n), x="count", y="side_effect", orientation="h",
                     color="count", color_continuous_scale="Teal")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500,
                           coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Side effects by therapeutic category")
        cat_pick = st.selectbox("Category", sorted(df["condition_category"].unique()))
        sub = df[df["condition_category"] == cat_pick]
        sub_freq = build_side_effect_frequency(sub)
        st.dataframe(sub_freq.head(10), use_container_width=True, hide_index=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🏆 Top drugs by review volume")
            top_reviewed = df.sort_values("num_reviews", ascending=False).head(10)
            fig = px.bar(top_reviewed, x="num_reviews", y="brand_name", orientation="h",
                         color="condition_category")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("⭐ Most effective drugs")
            most_effective = df.sort_values("effectiveness_score", ascending=False).head(10)
            fig = px.bar(most_effective, x="effectiveness_score", y="brand_name", orientation="h",
                         color="condition_category")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420,
                               xaxis_range=[0, 10])
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Effectiveness vs. price")
        fig = px.scatter(df, x="price_usd", y="effectiveness_score", color="condition_category",
                          size="num_reviews", hover_name="brand_name", log_x=True,
                          labels={"price_usd": "Monthly price (USD, log scale)",
                                  "effectiveness_score": "Effectiveness score"})
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Drugs per category")
            cat_counts = df["condition_category"].value_counts().reset_index()
            cat_counts.columns = ["category", "count"]
            fig = px.bar(cat_counts, x="count", y="category", orientation="h", color="count",
                         color_continuous_scale="Sunset")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=480,
                               coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Route of administration")
            route_counts = df["route"].value_counts().reset_index()
            route_counts.columns = ["route", "count"]
            fig = px.pie(route_counts, names="route", values="count", hole=0.4)
            fig.update_layout(height=480)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Dosage form distribution")
        form_counts = df["dosage_form"].value_counts().reset_index()
        form_counts.columns = ["dosage_form", "count"]
        fig = px.bar(form_counts, x="dosage_form", y="count", color="count",
                     color_continuous_scale="Blues")
        fig.update_layout(coloraxis_showscale=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Pregnancy safety distribution")
            preg_counts = df["pregnancy_category"].value_counts().reset_index()
            preg_counts.columns = ["pregnancy_category", "count"]
            order = ["Safe", "Consult", "Caution", "Dangerous"]
            preg_counts["pregnancy_category"] = pd.Categorical(preg_counts["pregnancy_category"],
                                                                 categories=order, ordered=True)
            preg_counts = preg_counts.sort_values("pregnancy_category")
            colors = {"Safe": "#16a34a", "Consult": "#0ea5e9", "Caution": "#f59e0b", "Dangerous": "#dc2626"}
            fig = px.bar(preg_counts, x="pregnancy_category", y="count",
                         color="pregnancy_category", color_discrete_map=colors)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("OTC vs. Prescription")
            otc_counts = df["otc_or_rx"].value_counts().reset_index()
            otc_counts.columns = ["type", "count"]
            fig = px.pie(otc_counts, names="type", values="count", hole=0.5,
                         color="type", color_discrete_map={"OTC": "#16a34a", "Rx": "#334155"})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Price distribution by category")
        fig = px.box(df, x="condition_category", y="price_usd", color="condition_category")
        fig.update_layout(showlegend=False, height=450, yaxis_title="Monthly price (USD)")
        st.plotly_chart(fig, use_container_width=True)

# ========================================================================================
# PAGE: SEARCH & LOOKUP
# ========================================================================================
elif page == "🔍 Search & Lookup":
    st.title("🔍 Drug Search & Lookup")
    st.caption("Filter the catalog like a formulary search engine.")

    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        query = c1.text_input("Search by name (brand / generic / substance)")
        categories = c2.multiselect("Condition category", sorted(df["condition_category"].unique()))
        routes = c3.multiselect("Route", sorted(df["route"].unique()))

        c4, c5, c6 = st.columns(3)
        forms = c4.multiselect("Dosage form", sorted(df["dosage_form"].unique()))
        otc_rx = c5.multiselect("OTC / Rx", sorted(df["otc_or_rx"].unique()))
        preg = c6.multiselect("Pregnancy safety", ["Safe", "Consult", "Caution", "Dangerous"])

        c7, c8 = st.columns(2)
        eff_range = c7.slider("Effectiveness score", 1.0, 10.0, (1.0, 10.0))
        price_range = c8.slider("Monthly price (USD)", float(df["price_usd"].min()),
                                 float(df["price_usd"].max()),
                                 (float(df["price_usd"].min()), float(df["price_usd"].max())))

    filtered = df.copy()
    if query:
        q = query.lower()
        mask = (filtered["brand_name"].str.lower().str.contains(q) |
                filtered["generic_name"].str.lower().str.contains(q) |
                filtered["substance_name"].str.lower().str.contains(q))
        filtered = filtered[mask]
    if categories:
        filtered = filtered[filtered["condition_category"].isin(categories)]
    if routes:
        filtered = filtered[filtered["route"].isin(routes)]
    if forms:
        filtered = filtered[filtered["dosage_form"].isin(forms)]
    if otc_rx:
        filtered = filtered[filtered["otc_or_rx"].isin(otc_rx)]
    if preg:
        filtered = filtered[filtered["pregnancy_category"].isin(preg)]
    filtered = filtered[filtered["effectiveness_score"].between(*eff_range)]
    filtered = filtered[filtered["price_usd"].between(*price_range)]

    st.markdown(f"**{len(filtered)}** drugs match your filters.")
    st.dataframe(
        filtered[["brand_name", "generic_name", "condition_category", "route", "dosage_form",
                  "otc_or_rx", "effectiveness_score", "price_usd", "pregnancy_category"]]
        .sort_values("effectiveness_score", ascending=False),
        use_container_width=True, hide_index=True, height=320,
    )

    st.divider()
    st.subheader("Drug detail")
    if len(filtered):
        pick = st.selectbox("Select a drug to inspect", filtered["brand_name"].tolist())
        row = filtered[filtered["brand_name"] == pick].iloc[0]
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"### {row['brand_name']}")
            st.markdown(f"*{row['generic_name']}* ({row['substance_name']})")
            st.markdown(f"<span class='pill'>{row['condition_category']}</span> "
                        f"<span class='pill'>{row['route']}</span> "
                        f"<span class='pill'>{row['dosage_form']}</span> "
                        f"<span class='pill'>{row['otc_or_rx']}</span>", unsafe_allow_html=True)
            st.metric("Effectiveness", f"{row['effectiveness_score']}/10")
            st.metric("Monthly price", f"${row['price_usd']:.2f}")
            st.metric("Reviews", f"{int(row['num_reviews']):,}")
        with c2:
            st.markdown(f"**Indicated for:** {row['condition']}")
            st.markdown(f"**Dosage:** {row['dosage']}")
            st.markdown(f"**Side effects:** {row['side_effects']}")
            st.markdown(f"**Warnings:** {row['warnings']}")
            st.markdown(f"**Contraindications:** {row['contraindications']}")
            st.markdown(f"**Drug interactions:** {row['drug_interactions']}")
            st.markdown(f"**Pregnancy:** {row['pregnancy_category']} — {row['pregnancy_warning']}")
            st.markdown(f"**Pediatric use:** {row['pediatric_use']}")
            st.markdown(f"**Geriatric use:** {row['geriatric_use']}")
    else:
        st.info("No drugs match the current filters.")

# ========================================================================================
# PAGE: COMPARE DRUGS
# ========================================================================================
elif page == "⚖️ Compare Drugs":
    st.title("⚖️ Compare Drugs")
    st.caption("Pick 2–4 drugs to compare side by side.")

    picks = st.multiselect("Choose drugs", df["brand_name"].tolist(),
                            default=df["brand_name"].tolist()[:2], max_selections=4)

    if len(picks) < 2:
        st.info("Select at least 2 drugs to compare.")
    else:
        sub = df[df["brand_name"].isin(picks)].set_index("brand_name").loc[picks]

        st.subheader("Side-by-side comparison")
        compare_fields = ["generic_name", "condition_category", "route", "dosage_form",
                           "otc_or_rx", "dosage", "effectiveness_score", "price_usd",
                           "num_reviews", "pregnancy_category", "side_effects", "warnings",
                           "drug_interactions"]
        display_tbl = sub[compare_fields].T
        display_tbl.index = ["Generic name", "Category", "Route", "Dosage form", "OTC/Rx",
                              "Dosage", "Effectiveness (0-10)", "Monthly price ($)", "Reviews",
                              "Pregnancy safety", "Side effects", "Warnings", "Interactions"]
        st.dataframe(display_tbl, use_container_width=True)

        st.subheader("Effectiveness vs. price")
        fig = px.bar(sub.reset_index(), x="brand_name", y="effectiveness_score",
                     color="brand_name", text="effectiveness_score")
        fig.update_layout(showlegend=False, yaxis_range=[0, 10], height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Multi-attribute radar")
        radar_metrics = ["effectiveness_score", "num_reviews", "half_life_hours"]
        radar_df = sub[radar_metrics].copy()
        # normalize 0-1 per metric so scales are comparable on the radar
        norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-9)
        fig = go.Figure()
        for name in norm.index:
            fig.add_trace(go.Scatterpolar(
                r=list(norm.loc[name].values) + [norm.loc[name].values[0]],
                theta=["Effectiveness", "Review volume", "Half-life"] + ["Effectiveness"],
                fill="toself", name=name,
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Values are normalized (0–1) across the selected drugs for visual comparison only.")

# ========================================================================================
# PAGE: AI CATEGORY CLASSIFIER
# ========================================================================================
elif page == "🤖 AI Category Classifier":
    st.title("🤖 AI Therapeutic Category Classifier")
    st.caption("Runs the packaged CatBoost model (drug_classifier_package.pkl) on a drug label you describe.")
    st.markdown(GLOBAL_DISCLAIMER)

    if model_error:
        st.error(f"Model unavailable: {model_error}")
    else:
        st.write("Fill in as much label detail as you have — the model was trained to read "
                 "these free-text fields plus route/dosage form.")

        preset = st.selectbox("Prefill from an existing catalog entry (optional)",
                               ["-- blank --"] + df["brand_name"].tolist())

        if preset != "-- blank --":
            r = df[df["brand_name"] == preset].iloc[0]
            defaults = dict(
                brand_name=r["brand_name"], generic_name=r["generic_name"],
                substance_name=r["substance_name"], contraindications=r["contraindications"],
                pregnancy_warning=r["pregnancy_warning"], warnings=r["warnings"],
                side_effects=r["side_effects"], drug_interactions=r["drug_interactions"],
                dosage=r["dosage"], pediatric_use=r["pediatric_use"],
                geriatric_use=r["geriatric_use"], effectiveness=r["effectiveness"],
                route=r["route"], dosage_form=r["dosage_form"],
            )
        else:
            defaults = {k: "" for k in [
                "brand_name", "generic_name", "substance_name", "contraindications",
                "pregnancy_warning", "warnings", "side_effects", "drug_interactions",
                "dosage", "pediatric_use", "geriatric_use", "effectiveness"]}
            defaults["route"] = "oral"
            defaults["dosage_form"] = "tablet"

        with st.form("classifier_form"):
            c1, c2 = st.columns(2)
            with c1:
                brand_name = st.text_input("Brand name", defaults["brand_name"])
                generic_name = st.text_input("Generic name", defaults["generic_name"])
                substance_name = st.text_input("Substance name", defaults["substance_name"])
                dosage = st.text_input("Dosage", defaults["dosage"])
                route = st.selectbox("Route", sorted(df["route"].unique()),
                                      index=sorted(df["route"].unique()).index(defaults["route"])
                                      if defaults["route"] in df["route"].unique() else 0)
                dosage_form = st.selectbox("Dosage form", sorted(df["dosage_form"].unique()),
                                            index=sorted(df["dosage_form"].unique()).index(defaults["dosage_form"])
                                            if defaults["dosage_form"] in df["dosage_form"].unique() else 0)
            with c2:
                side_effects = st.text_area("Side effects", defaults["side_effects"], height=80)
                warnings_txt = st.text_area("Warnings", defaults["warnings"], height=80)
                contraindications = st.text_area("Contraindications", defaults["contraindications"], height=80)
                drug_interactions = st.text_area("Drug interactions", defaults["drug_interactions"], height=80)

            c3, c4, c5 = st.columns(3)
            pregnancy_warning = c3.text_area("Pregnancy warning", defaults["pregnancy_warning"], height=80)
            pediatric_use = c4.text_area("Pediatric use", defaults["pediatric_use"], height=80)
            geriatric_use = c5.text_area("Geriatric use", defaults["geriatric_use"], height=80)
            effectiveness = st.text_input("Effectiveness note", defaults["effectiveness"])

            submitted = st.form_submit_button("🔮 Predict therapeutic category", type="primary")

        if submitted:
            record = dict(
                brand_name=brand_name, generic_name=generic_name, substance_name=substance_name,
                contraindications=contraindications, pregnancy_warning=pregnancy_warning,
                warnings=warnings_txt, side_effects=side_effects, drug_interactions=drug_interactions,
                dosage=dosage, pediatric_use=pediatric_use, geriatric_use=geriatric_use,
                effectiveness=effectiveness, route=route, dosage_form=dosage_form,
            )
            try:
                label, proba_map = predict_condition_category(record, model_package)
                st.success(f"Predicted category: **{label}**")

                proba_df = pd.DataFrame(list(proba_map.items()), columns=["category", "probability"])
                fig = px.bar(proba_df.head(10), x="probability", y="category", orientation="h",
                             color="probability", color_continuous_scale="Teal")
                fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420,
                                   coloraxis_showscale=False, xaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:  # noqa: BLE001
                st.error(f"Prediction failed: {e}")

# ========================================================================================
# PAGE: SIMILAR DRUG FINDER
# ========================================================================================
elif page == "🧬 Similar Drug Finder":
    st.title("🧬 Similar Drug / Alternative Finder")
    st.caption("Content-based similarity over condition, side-effects and warnings text — "
               "useful for surfacing potential therapeutic alternatives to discuss with a pharmacist.")
    st.markdown(GLOBAL_DISCLAIMER)

    anchor = st.selectbox("Find drugs similar to:", df["brand_name"].tolist())
    top_k = st.slider("Number of similar drugs", 3, 15, 6)

    vec, matrix = build_similarity_index(tuple(df["drug_id"]), df)
    idx = df.index[df["brand_name"] == anchor][0]
    row_pos = df.index.get_loc(idx)
    sims = cosine_similarity(matrix[row_pos], matrix).flatten()
    sim_df = df.copy()
    sim_df["similarity"] = sims
    sim_df = sim_df[sim_df["brand_name"] != anchor].sort_values("similarity", ascending=False).head(top_k)

    st.dataframe(
        sim_df[["brand_name", "generic_name", "condition_category", "similarity",
                "effectiveness_score", "price_usd", "side_effects"]]
        .assign(similarity=lambda d: (d["similarity"] * 100).round(1)),
        use_container_width=True, hide_index=True,
    )

    fig = px.bar(sim_df, x="similarity", y="brand_name", orientation="h", color="condition_category")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380,
                       xaxis_title="Similarity score")
    st.plotly_chart(fig, use_container_width=True)

# ========================================================================================
# PAGE: INTERACTION CHECKER
# ========================================================================================
elif page == "🧪 Interaction Checker":
    st.title("🧪 Drug Interaction Checker")
    st.caption("Keyword-based check across each drug's `drug_interactions` label text — "
               "a starting point, not a substitute for a full interaction database.")
    st.markdown(GLOBAL_DISCLAIMER)

    picks = st.multiselect("Select the drugs the patient is taking", df["brand_name"].tolist())

    if len(picks) < 1:
        st.info("Select at least one drug.")
    else:
        sub = df[df["brand_name"].isin(picks)]
        st.subheader("Reported interaction substances per drug")
        for _, r in sub.iterrows():
            st.markdown(f"**{r['brand_name']}** ({r['generic_name']}): {r['drug_interactions']}")

        if len(picks) >= 2:
            st.subheader("Cross-drug flag check")
            names_lower = [n.lower() for n in sub["generic_name"]] + [n.lower() for n in sub["brand_name"]]
            flags = []
            for _, r in sub.iterrows():
                text = str(r["drug_interactions"]).lower()
                for other in sub["brand_name"]:
                    if other == r["brand_name"]:
                        continue
                    other_generic = df[df["brand_name"] == other]["generic_name"].iloc[0].lower()
                    if other_generic in text or other.lower() in text:
                        flags.append((r["brand_name"], other))
            if flags:
                st.error("⚠️ Potential interaction(s) flagged:")
                for a, b in flags:
                    st.markdown(f"- **{a}** ↔ **{b}**")
            else:
                st.success("No direct name-matches found between the selected drugs' interaction text. "
                           "This does NOT rule out an interaction — consult a pharmacist for a full check.")

        st.subheader("Common interacting substance classes to ask about")
        common = ["alcohol", "grapefruit juice", "warfarin", "aspirin", "MAO inhibitors", "St. John's Wort"]
        st.write(", ".join(f"`{c}`" for c in common))

# ========================================================================================
# PAGE: CALCULATORS & TOOLS
# ========================================================================================
elif page == "🧮 Calculators & Tools":
    st.title("🧮 Calculators & Tools")

    tool = st.selectbox("Choose a tool", [
        "BMI Calculator",
        "Dose ↔ Concentration (mL to draw up)",
        "Dilution Calculator (C1V1 = C2V2)",
        "Weight-Based / Pediatric Dosing",
        "Creatinine Clearance (Cockcroft-Gault)",
        "Unit Converter",
    ])

    if tool == "BMI Calculator":
        st.subheader("BMI Calculator")
        c1, c2, c3 = st.columns(3)
        unit_sys = c1.radio("Units", ["Metric (kg/cm)", "Imperial (lb/in)"])
        if unit_sys.startswith("Metric"):
            weight = c2.number_input("Weight (kg)", 1.0, 400.0, 70.0)
            height = c3.number_input("Height (cm)", 50.0, 250.0, 170.0)
        else:
            weight_lb = c2.number_input("Weight (lb)", 2.0, 900.0, 154.0)
            height_in = c3.number_input("Height (in)", 20.0, 100.0, 67.0)
            weight = calc.lbs_to_kg(weight_lb)
            height = calc.in_to_cm(height_in)

        if st.button("Calculate BMI", type="primary"):
            value = calc.bmi(weight, height)
            category = calc.bmi_category(value)
            c1, c2 = st.columns(2)
            c1.metric("BMI", value)
            c2.metric("Category", category)
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=value,
                gauge={"axis": {"range": [10, 45]},
                       "steps": [
                           {"range": [10, 18.5], "color": "#93c5fd"},
                           {"range": [18.5, 25], "color": "#86efac"},
                           {"range": [25, 30], "color": "#fde68a"},
                           {"range": [30, 45], "color": "#fca5a5"},
                       ],
                       "bar": {"color": "#1e293b"}}))
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("BMI is a screening tool only and does not directly measure body fat or individual health.")

    elif tool == "Dose ↔ Concentration (mL to draw up)":
        st.subheader("Dose from Concentration")
        st.caption("For liquid/injectable formulations. E.g. desired 250mg dose at a 50mg/mL concentration → 5 mL.")
        c1, c2 = st.columns(2)
        dose = c1.number_input("Desired dose (mg)", 0.0, 10000.0, 250.0)
        conc = c2.number_input("Concentration (mg/mL)", 0.01, 1000.0, 50.0)
        if st.button("Calculate volume", type="primary"):
            vol = calc.dose_from_concentration(dose, conc)
            st.metric("Volume to administer", f"{vol} mL")

        st.divider()
        st.caption("Or pick a liquid/injectable drug from the catalog to auto-fill concentration:")
        liquid_df = df[df["concentration_mg_per_ml"].notna()]
        if len(liquid_df):
            pick = st.selectbox("Liquid/injectable drug", liquid_df["brand_name"].tolist())
            row = liquid_df[liquid_df["brand_name"] == pick].iloc[0]
            st.write(f"**{pick}** concentration: {row['concentration_mg_per_ml']} mg/mL "
                     f"(strength label: {row['dosage']})")

    elif tool == "Dilution Calculator (C1V1 = C2V2)":
        st.subheader("Dilution Calculator")
        st.caption("Solve C1×V1 = C2×V2 — leave the value you want solved for empty (set to 0 and uncheck 'known').")
        c1, c2, c3, c4 = st.columns(4)
        know = {}
        vals = {}
        for label, col, key in [("C1 (stock conc.)", c1, "c1"), ("V1 (stock vol.)", c2, "v1"),
                                 ("C2 (target conc.)", c3, "c2"), ("V2 (target vol.)", c4, "v2")]:
            with col:
                known = st.checkbox(f"Known: {label}", value=True, key=f"chk_{key}")
                val = st.number_input(label, 0.0, 100000.0, 100.0, key=f"val_{key}", disabled=not known)
                know[key] = known
                vals[key] = val if known else None

        unknowns = [k for k, v in know.items() if not v]
        if len(unknowns) != 1:
            st.warning("Uncheck exactly ONE field — that's the value that will be solved for.")
        else:
            if st.button("Solve", type="primary"):
                result = calc.dilution_c1v1(**vals)
                st.success(f"**{unknowns[0].upper()} = {result}**")

    elif tool == "Weight-Based / Pediatric Dosing":
        st.subheader("Weight-Based Dosing")
        c1, c2, c3 = st.columns(3)
        weight = c1.number_input("Patient weight (kg)", 0.5, 300.0, 20.0)
        mg_per_kg = c2.number_input("Dose (mg/kg)", 0.01, 100.0, 10.0)
        max_dose = c3.number_input("Max dose cap (mg, 0 = no cap)", 0.0, 5000.0, 0.0)
        if st.button("Calculate dose", type="primary"):
            dose = calc.weight_based_dose(weight, mg_per_kg, max_dose if max_dose > 0 else None)
            st.metric("Calculated dose", f"{dose} mg")
            if max_dose > 0 and weight * mg_per_kg > max_dose:
                st.info(f"Capped at the maximum dose of {max_dose} mg (uncapped would be {weight*mg_per_kg:.1f} mg).")

    elif tool == "Creatinine Clearance (Cockcroft-Gault)":
        st.subheader("Estimated Creatinine Clearance (Cockcroft-Gault)")
        st.caption("A common pharmacist reference for renal dose adjustment.")
        c1, c2, c3, c4 = st.columns(4)
        age = c1.number_input("Age (years)", 1, 120, 55)
        weight = c2.number_input("Weight (kg)", 1.0, 300.0, 75.0)
        scr = c3.number_input("Serum creatinine (mg/dL)", 0.1, 20.0, 1.0)
        sex = c4.selectbox("Sex", ["Male", "Female"])
        if st.button("Calculate CrCl", type="primary"):
            inp = calc.CreatinineClearanceInput(age, weight, scr, sex == "Female")
            crcl = calc.cockcroft_gault_crcl(inp)
            st.metric("Estimated CrCl", f"{crcl} mL/min")
            band = ("Normal" if crcl >= 90 else "Mild impairment" if crcl >= 60 else
                    "Moderate impairment" if crcl >= 30 else "Severe impairment")
            st.info(f"Approximate renal function band: **{band}**")

    elif tool == "Unit Converter":
        st.subheader("Unit Converter")
        conv_type = st.radio("Type", ["Mass", "Volume"], horizontal=True)
        if conv_type == "Mass":
            units = ["mcg", "mg", "g", "kg"]
            c1, c2, c3 = st.columns(3)
            val = c1.number_input("Value", 0.0, 1e9, 500.0)
            f_unit = c2.selectbox("From", units, index=1)
            t_unit = c3.selectbox("To", units, index=0)
            st.metric("Result", f"{calc.convert_mass(val, f_unit, t_unit)} {t_unit}")
        else:
            units = ["mL", "L", "fl_oz", "tsp", "tbsp"]
            c1, c2, c3 = st.columns(3)
            val = c1.number_input("Value", 0.0, 1e9, 5.0)
            f_unit = c2.selectbox("From", units, index=3)
            t_unit = c3.selectbox("To", units, index=0)
            st.metric("Result", f"{calc.convert_volume(val, f_unit, t_unit)} {t_unit}")

# ========================================================================================
# PAGE: MEDICATION TRACKER
# ========================================================================================
elif page == "📅 Medication Tracker":
    st.title("📅 Medication Tracker")
    st.caption("Build a simple daily dosing schedule. Session-only (resets on refresh) — "
               "for a permanent tracker, connect this page to a database.")

    if "med_schedule" not in st.session_state:
        st.session_state["med_schedule"] = []

    with st.form("add_med"):
        c1, c2, c3, c4 = st.columns(4)
        drug_pick = c1.selectbox("Drug", df["brand_name"].tolist())
        times = c2.multiselect("Time(s) of day", ["Morning", "Noon", "Evening", "Bedtime"], default=["Morning"])
        dose_note = c3.text_input("Dose note (optional)", "")
        with_food = c4.selectbox("With food?", ["Either", "With food", "Empty stomach"])
        add = st.form_submit_button("➕ Add to schedule")
        if add and times:
            row = df[df["brand_name"] == drug_pick].iloc[0]
            for t in times:
                st.session_state["med_schedule"].append({
                    "Drug": drug_pick, "Generic": row["generic_name"], "Time": t,
                    "Dose note": dose_note or row["dosage"], "Food": with_food,
                })

    if st.session_state["med_schedule"]:
        sched_df = pd.DataFrame(st.session_state["med_schedule"])
        order = ["Morning", "Noon", "Evening", "Bedtime"]
        sched_df["Time"] = pd.Categorical(sched_df["Time"], categories=order, ordered=True)
        sched_df = sched_df.sort_values("Time")
        st.subheader("Today's schedule")
        for t in order:
            block = sched_df[sched_df["Time"] == t]
            if len(block):
                st.markdown(f"**{t}**")
                st.dataframe(block[["Drug", "Generic", "Dose note", "Food"]],
                             use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear schedule"):
            st.session_state["med_schedule"] = []
            st.rerun()
    else:
        st.info("No medications added yet.")

# ========================================================================================
# PAGE: ABOUT
# ========================================================================================
elif page == "ℹ️ About This Model":
    st.title("ℹ️ About the Model & Data")

    st.subheader("Model")
    st.markdown("""
- **Algorithm:** CatBoost `MultiClass` classifier (14 therapeutic categories)
- **Target classes:** Autoimmune, Cancer, Cardiovascular, Dermatology, ENT, Endocrine,
  Gastrointestinal, Infection, Neurological, Ophthalmology, Pain, Psychiatric, Renal, Respiratory
- **Feature pipeline:** TF-IDF (1-2 grams, 5000 features) on concatenated label text
  → `SelectKBest` (mutual information, k=2000) → one-hot encoded `route`/`dosage_form`
  → `TruncatedSVD` (200 components) → `StandardScaler` → CatBoost
- **Hyperparameters:** tuned via Optuna (20 trials), balanced class weights, early stopping
- **Training data:** a schema-matched **synthetic** dataset (the real openFDA extract used at
  training time was unavailable, so the notebook falls back to a generator matching the same
  columns). Treat predictions as illustrative rather than clinically validated.
    """)

    if model_error:
        st.error(f"Model status: not loaded ({model_error})")
    else:
        st.success("Model status: loaded and ready ✓")
        with st.expander("Class list"):
            st.write(list(model_package["target_classes"]))

    st.subheader("Demo drug catalog")
    st.markdown("""
The Analytics, Search, Compare, Similar-Drug-Finder and Interaction Checker pages run on a
**420-row synthetic catalog** generated with the same schema the model expects, so every page
works out of the box. Pharmacists/analysts can upload a real catalog CSV from the sidebar
(**Use your own drug catalog**) with the required columns — the whole dashboard will switch
to using it automatically.
    """)
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    st.markdown(GLOBAL_DISCLAIMER)
