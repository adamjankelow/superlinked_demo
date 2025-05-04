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
)

from backend.types import (
    SearchInputs,
    WeightedParams,
    NumericParams,
    CombinedParams,
)


@st.cache_resource
def build_context():
    data_frame = load_data()
    app, index, food_item, desc_space, cat_text_space, cat_cat_space, cal_space = build_superlinked_app(data_frame)
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


# ── Simple ──────────────────────────────────────────────────────────────
if mode == "Simple":
    with st.form("simple_search_form"):
        description_query = st.text_input("Food description", "cereal with sugar")
        submitted = st.form_submit_button("🔍 Search")

    if submitted:
        inputs = SearchInputs(description_query=description_query)
        st.dataframe(simple_search(ctx, inputs))

# ── Weighted ────────────────────────────────────────────────────────────
elif mode == "Weighted":
    with st.form("weighted_search_form"):
        description_query = st.text_input("Food description", "apple")
        category_query = st.text_input("Food category", "dessert")
        desc_weight = st.slider("Description weight", -3.0, 3.0, 1.0)
        cat_weight = st.slider("Category weight", -3.0, 3.0, 1.0)
        submitted = st.form_submit_button("🔍 Search")

    if submitted:
        inputs = SearchInputs(description_query, category_query)
        params = WeightedParams(desc_weight, cat_weight)
        results = weighted_search(ctx, inputs, params)
        st.dataframe(results)

        top10_ids = results.nlargest(10, "similarity_score").id.astype(int).tolist()
        umap_df_top10 = load_umap_df().loc[top10_ids]
        st.write("#### UMAP Visualization of Top-10 Results")
        fig = plot_umap_scatter(umap_df_top10)
        st.pyplot(fig)

# ── Numeric ─────────────────────────────────────────────────────────────
elif mode == "Numeric":
    with st.form("numeric_search_form"):
        description_query = st.text_input("Food description", "chicken")
        calories_val = st.number_input("Calories per 100 g", 0, 1000, 200)
        desc_weight = st.slider("Description weight", -3.0, 3.0, 1.0)
        cal_weight = st.slider("Calories weight", -3.0, 3.0, 1.0)
        submitted = st.form_submit_button("🔍 Search")

    if submitted:
        inputs = SearchInputs(description_query=description_query, calories_val=calories_val)
        params = NumericParams(desc_weight, cal_weight)
        top10, mean_calories = numeric_search(ctx, inputs, params)
        st.dataframe(top10)
        st.bar_chart(top10.set_index("description")["calories"])
        st.write(f"Mean calories (top-10): **{mean_calories:.1f}**")

# ── Combined ────────────────────────────────────────────────────────────
else:  # mode == "Combined"
    category_choices = sorted(data_frame.food_category.unique())
    with st.form("combined_search_form"):
        description_query = st.text_input("Food description")
        category_query = st.selectbox("Food category filter", category_choices)
        calories_val = st.number_input("Calories per 100 g", 0, 1000)
        desc_weight = st.slider("Description weight", -3.0, 3.0, 1.0)
        cal_weight = st.slider("Calories weight", -3.0, 3.0, 1.0)
        submitted = st.form_submit_button("🔍 Search")

    if submitted:
        inputs = SearchInputs(
            description_query=description_query,
            category_query=category_query,
            calories_val=calories_val,
        )
        params = CombinedParams(desc_weight, cal_weight)
        st.dataframe(combined_search(ctx, inputs, params))

# streamlit run src/frontend/main.py