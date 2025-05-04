# frontend/main.py
import streamlit as st
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# data helpers
from backend.utils.data import (
    load_data,
    build_superlinked_app,
    plot_umap_scatter,
)
from backend.utils.umap import load_umap_df

from backend.queries import (
    simple_search,
    weighted_search,
    numeric_search,
    combined_search,
    WeightedParams,
    NumericParams,
    CombinedParams,
)
@st.cache_data
def load_data_for_frontend():
    df = load_data()
    return df

st.set_page_config(page_title="Semantic Food Search", page_icon="🥦")
st.title("🥦 Semantic Search on Food Database")


df = load_data_for_frontend()
app, index, food_item, desc_space, cat_text_space, cat_cat_space, cal_space = (
    build_superlinked_app(df)
)

mode = st.sidebar.radio("Mode", ["Simple", "Weighted", "Numeric", "Combined"])

# --- simple ---
if mode == "Simple":
    q = st.text_input("Food description", "cereal with sugar")
    if q:
        st.dataframe(simple_search(app, index, food_item, desc_space, q))

# --- weighted ---
elif mode == "Weighted":
    q = st.text_input("Food description", "apple")
    cat = st.text_input("Food category", "dessert")
    dw = st.slider("Description weight", -3.0, 3.0, 1.0)
    cw = st.slider("Category weight", -3.0, 3.0, 1.0)
    if q and cat:
        res = weighted_search(
            app,
            index,
            food_item,
            desc_space,
            cat_text_space,
            cat_cat_space,
            q,
            cat,
            WeightedParams(dw, cw),
        )
        st.dataframe(res)
        # return the top 10 results
        top10 = res.sort_values("similarity_score", ascending=False).head(10)
        list_of_ids = top10.id.astype(int).tolist()
        umap_df = load_umap_df()    
        
        umap_df_top10 = umap_df[umap_df.index.isin(list_of_ids)]
        plot_umap_scatter(umap_df_top10)

# --- numeric ---
elif mode == "Numeric":
    q = st.text_input("Food description", "chicken")
    cal_val = st.number_input("Calories per 100 g", 0, 1000, 200)
    dw = st.slider("Description weight", -3.0, 3.0, 1.0)
    cw = st.slider("Calories weight", -3.0, 3.0, 1.0)
    if q:
        top10, mean_cal = numeric_search(
            app,
            index,
            food_item,
            desc_space,
            cal_space,
            q,
            cal_val,
            NumericParams(dw, cw),
        )
        st.dataframe(top10)
        st.bar_chart(top10.set_index("description")["calories"])
        st.write(f"Mean calories of top‑10: **{mean_cal:.1f}**")

# --- combined ---
else:
    categories = sorted(df.food_category.unique())
    cat_filter = st.selectbox("Food category filter", categories)
    q = st.text_input("Food description")
    cal_val = st.number_input("Calories per 100 g", 0, 1000)
    dw = st.slider("Description weight", -3.0, 3.0, 1.0, key="cdw")
    cw = st.slider("Calories weight", -3.0, 3.0, 1.0, key="ccw")
    if q:
        res = combined_search(
            app,
            index,
            food_item,
            desc_space,
            cat_cat_space,
            cal_space,
            q,
            cat_filter,
            cal_val,
            CombinedParams(dw, cw),
        )
        st.dataframe(res)


# streamlit run src/frontend/main.py