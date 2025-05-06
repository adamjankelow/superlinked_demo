import streamlit as st
st.set_page_config(page_title="Semantic Food Search", page_icon="🥦")

from backend.ingest.loader import load_data, build_superlinked_app
from backend.features.umap import plot_umap_scatter, subset_top_n_umap
from backend.search.queries import (
    simple_search,
    weighted_search,
    numeric_search,
    combined_search,
)
from backend.search.types import (
    SearchCtx,
    SearchInputs,
    SearchWeights
)

from backend.config import settings

# ───────────────────────── setup ─────────────────────────

@st.cache_data()
def get_df():
    """Load the food database from a Parquet file."""
    return load_data()

@st.cache_resource
def build_context():
    df = load_data()
    ctx = build_superlinked_app(df)
    return df, ctx

# @st.cache_data(show_spinner=False)
# def get_umap():
#     return load_umap_df()

df, ctx = build_context()

st.title("🥦 Semantic Search on Food Database")
mode = st.sidebar.radio("Search Mode", ["Simple", "Weighted", "Numeric", "Combined"])


# ───────────────────────── UI Components ─────────────────────────

def render_simple_ui(ctx: SearchCtx):
    """Render the UI for simple search mode."""
    with st.form("simple_search_form"):
        q = st.text_input("Food description", "cereal with sugar")
        submitted = st.form_submit_button("🔍 Search")
    if submitted:
        inputs = SearchInputs(description_query=q)
        st.dataframe(simple_search(ctx, inputs))


def render_weighted_ui(ctx: SearchCtx, df):
    """Render the UI for weighted search mode."""
    with st.form("weighted_search_form"):
        q = st.text_input("Food description", "apple")
        cat = st.text_input("Food category", "dessert")
        dw = st.slider("Description weight", -3.0, 3.0, 1.0)
        cw = st.slider("Category weight",    -3.0, 3.0, 1.0)
        submitted = st.form_submit_button("🔍 Search")
    if submitted:
        inputs = SearchInputs(description_query=q, category_query=cat)
        params = SearchWeights(desc_weight=dw, cat_weight=cw)
        results = weighted_search(ctx, inputs, params)
        st.dataframe(results)
        umap_df_top10 = subset_top_n_umap(results, top_n=10)
        st.write("#### UMAP Visualization of Top-10 Results")
        st.pyplot(plot_umap_scatter(umap_df_top10))


def render_numeric_ui(ctx: SearchCtx):
    """Render the UI for numeric search mode."""
    with st.form("numeric_search_form"):
        q = st.text_input("Food description", "chicken")
        cal = st.number_input("Calories per 100 g", min_value=settings.calories_min, max_value=settings.calories_max, value=200)
        dw = st.slider("Description weight", min_value=-3.0, max_value=3.0, value=1.0)
        cw = st.slider("Calories weight",    min_value=-3.0, max_value=3.0, value=1.0)
        submitted = st.form_submit_button("🔍 Search")
    if submitted:
        inputs = SearchInputs(description_query=q, calories_val=cal)
        params = SearchWeights(desc_weight=dw, cal_weight=cw)
        top10, mean_cal = numeric_search(ctx, inputs, params)
        st.dataframe(top10)
        st.bar_chart(top10.set_index("description")["calories"])
        st.write(f"Mean calories (top-10): **{mean_cal:.1f}**")


def render_combined_ui(ctx: SearchCtx, df):
    """Render the UI for combined search mode."""
    cats = sorted(df.food_category.unique())
    with st.form("combined_search_form"):
        cat_filter = st.selectbox("Food category filter", cats)
        q = st.text_input("Food description")
        cal = st.number_input("Calories per 100 g", min_value=settings.calories_min, max_value=settings.calories_max, value=200)
        dw = st.slider("Description weight", -3.0, 3.0, 1.0)
        cw = st.slider("Calories weight",    -3.0, 3.0, 1.0)
        submitted = st.form_submit_button("🔍 Search")
    if submitted:
        inputs = SearchInputs(description_query=q, category_query=cat_filter, calories_val=cal)
        params = SearchWeights(desc_weight=dw, cal_weight=cw)
        results = combined_search(ctx, inputs, params)
        st.dataframe(results)


# ───────────────────────── Dispatcher ─────────────────────────

if mode == "Simple":
    render_simple_ui(ctx)
elif mode == "Weighted":
    render_weighted_ui(ctx, df)
elif mode == "Numeric":
    render_numeric_ui(ctx)
else:
    render_combined_ui(ctx, df)

# To run: streamlit run src/frontend/main.py
