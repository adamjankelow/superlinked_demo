# frontend/main.py  ── Streamlit UI using SearchCtx with form-based searches
import streamlit as st
st.set_page_config(page_title="Semantic Food Search", page_icon="🥦")

from backend.utils.data import (
    load_data,
    build_superlinked_app,
)
from backend.utils.umap import (
    load_umap_df,
    plot_umap_scatter,
)
from backend.queries import (
    SearchCtx,
    simple_search,
    weighted_search,
    numeric_search,
    combined_search,
    WeightedParams,
    NumericParams,
    CombinedParams,
)


@st.cache_resource
def build_context():
    data_frame = load_data()
    app, index, food_item, desc_space, cat_text_space, cat_cat_space, cal_space = (
        build_superlinked_app(data_frame)
    )
    ctx = SearchCtx(
        app=app,
        index=index,
        food_item=food_item,
        desc_space=desc_space,
        cat_text_space=cat_text_space,
        cat_cat_space=cat_cat_space,
        cal_space=cal_space,
    )
    return data_frame, ctx


data_frame, ctx = build_context()

# ── UI ─────────────────────────────────────────────────────────────────

st.title("🥦 Semantic Search on Food Database")

mode = st.sidebar.radio("Mode", ["Simple", "Weighted", "Numeric", "Combined"])

# ── Simple ─────────────────────────────────────────────────────────────────
if mode == "Simple":
    with st.form("simple_search_form"):
        description_query = st.text_input("Food description", "cereal with sugar")
        submitted         = st.form_submit_button("🔍 Search")

    if submitted:
        st.dataframe(simple_search(ctx, description_query))


# ── Weighted ───────────────────────────────────────────────────────────────
elif mode == "Weighted":
    with st.form("weighted_search_form"):
        description_query  = st.text_input("Food description", "apple")
        category_query     = st.text_input("Food category", "dessert")
        description_weight = st.slider("Description weight", -3.0, 3.0, 1.0)
        category_weight    = st.slider("Category weight",    -3.0, 3.0, 1.0)
        submitted          = st.form_submit_button("🔍 Search")

    if submitted:
        results = weighted_search(
            ctx,
            description_query,
            category_query,
            WeightedParams(description_weight, category_weight),
        )
        st.dataframe(results)

        # UMAP for top-10
        top10_ids      = results.nlargest(10, "similarity_score").id.astype(int).tolist()
        umap_df_top10  = load_umap_df().loc[top10_ids]
        st.write("#### UMAP Visualization of Top-10 Results")
        fig = plot_umap_scatter(umap_df_top10)
        st.pyplot(fig)


# ── Numeric ────────────────────────────────────────────────────────────────
elif mode == "Numeric":
    with st.form("numeric_search_form"):
        description_query  = st.text_input("Food description", "chicken")
        calories_target    = st.number_input("Calories per 100 g", 0, 1000, 200)
        description_weight = st.slider("Description weight", -3.0, 3.0, 1.0)
        calories_weight    = st.slider("Calories weight",    -3.0, 3.0, 1.0)
        submitted          = st.form_submit_button("🔍 Search")

    if submitted:
        top10, mean_calories = numeric_search(
            ctx,
            description_query,
            calories_target,
            NumericParams(description_weight, calories_weight),
        )
        st.dataframe(top10)
        st.bar_chart(top10.set_index("description")["calories"])
        st.write(f"Mean calories (top-10): **{mean_calories:.1f}**")


# ── Combined ───────────────────────────────────────────────────────────────
else:  # mode == "Combined"
    category_choices = sorted(data_frame.food_category.unique())
    with st.form("combined_search_form"):
        category_filter    = st.selectbox("Food category filter", category_choices)
        description_query  = st.text_input("Food description")
        calories_target    = st.number_input("Calories per 100 g", 0, 1000)
        description_weight = st.slider("Description weight", -3.0, 3.0, 1.0)
        calories_weight    = st.slider("Calories weight",    -3.0, 3.0, 1.0)
        submitted          = st.form_submit_button("🔍 Search")

    if submitted:
        results = combined_search(
            ctx,
            description_query,
            category_filter,
            calories_target,
            CombinedParams(description_weight, calories_weight),
        )
        st.dataframe(results)

# streamlit run src/frontend/main.py