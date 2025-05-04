import streamlit as st
st.set_page_config(page_title="Semantic Food Search", page_icon="🥦")

from backend.utils.data import load_data, build_superlinked_app
from backend.utils.umap import load_umap_df, plot_umap_scatter
from backend.queries import (
    simple_search,
    weighted_search,
    numeric_search,
    combined_search,
)
from backend.types import (
    SearchCtx,
    SearchInputs,
    WeightedParams,
    NumericParams,
    CombinedParams,
)

# ───────────────────────── setup ─────────────────────────

@st.cache_resource
def build_context():
    df = load_data()
    app, index, food_item, desc_space, cat_text_space, cat_cat_space, cal_space = build_superlinked_app(df)
    ctx = SearchCtx(app, index, food_item, desc_space, cat_text_space, cat_cat_space, cal_space)
    return df, ctx

df, ctx = build_context()

st.title("🥦 Semantic Search on Food Database")
mode = st.sidebar.radio("Search Mode", ["Simple", "Weighted", "Numeric", "Combined"])


# ───────────────────────── UI Components ─────────────────────────

def render_simple_ui(ctx: SearchCtx):
    with st.form("simple_search_form"):
        q = st.text_input("Food description", "cereal with sugar")
        submitted = st.form_submit_button("🔍 Search")
    if submitted:
        inputs = SearchInputs(description_query=q)
        st.dataframe(simple_search(ctx, inputs))


def render_weighted_ui(ctx: SearchCtx, df):
    with st.form("weighted_search_form"):
        q = st.text_input("Food description", "apple")
        cat = st.text_input("Food category", "dessert")
        dw = st.slider("Description weight", -3.0, 3.0, 1.0)
        cw = st.slider("Category weight",    -3.0, 3.0, 1.0)
        submitted = st.form_submit_button("🔍 Search")
    if submitted:
        inputs = SearchInputs(description_query=q, category_query=cat)
        params = WeightedParams(dw, cw)
        results = weighted_search(ctx, inputs, params)
        st.dataframe(results)
        top10_ids = results.nlargest(10, "similarity_score").id.astype(int).tolist()
        umap_df_top10 = load_umap_df().loc[top10_ids]
        st.write("#### UMAP Visualization of Top-10 Results")
        st.pyplot(plot_umap_scatter(umap_df_top10))


def render_numeric_ui(ctx: SearchCtx):
    with st.form("numeric_search_form"):
        q = st.text_input("Food description", "chicken")
        cal = st.number_input("Calories per 100 g", 0, 1000, 200)
        dw = st.slider("Description weight", -3.0, 3.0, 1.0)
        cw = st.slider("Calories weight",    -3.0, 3.0, 1.0)
        submitted = st.form_submit_button("🔍 Search")
    if submitted:
        inputs = SearchInputs(description_query=q, calories_val=cal)
        params = NumericParams(dw, cw)
        top10, mean_cal = numeric_search(ctx, inputs, params)
        st.dataframe(top10)
        st.bar_chart(top10.set_index("description")["calories"])
        st.write(f"Mean calories (top-10): **{mean_cal:.1f}**")


def render_combined_ui(ctx: SearchCtx, df):
    cats = sorted(df.food_category.unique())
    with st.form("combined_search_form"):
        cat_filter = st.selectbox("Food category filter", cats)
        q = st.text_input("Food description")
        cal = st.number_input("Calories per 100 g", 0, 1000)
        dw = st.slider("Description weight", -3.0, 3.0, 1.0)
        cw = st.slider("Calories weight",    -3.0, 3.0, 1.0)
        submitted = st.form_submit_button("🔍 Search")
    if submitted:
        inputs = SearchInputs(description_query=q, category_query=cat_filter, calories_val=cal)
        params = CombinedParams(dw, cw)
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
