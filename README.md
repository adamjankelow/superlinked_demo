# Semantic Food Search Demo

A demo semantic search engine for food items built with Superlinked and Streamlit. This demo showcases advanced search capabilities including semantic similarity, multi-criteria weighted search, and numeric range filtering.

## Features

### Simple Semantic Search
- Natural language search for food items based on descriptions
- Returns semantically similar results based off text


### Multi-Criteria Weighted Search  
- Combine text descriptions and food categories
- Adjust weights to fine-tune search relevance
- Interactive visualization of search results using UMAP
- See how different weights affect the results in real-time

### Numeric Range Search
- Filter by nutritional values like calories
- Combine numeric and semantic criteria
- Visual analysis of results with interactive charts
- Calculate statistics across search results

### Combined Search
- Unified interface for all search modes
- Filtering by food categories
- Flexible combination of search criteria
- Optimized for intuitive exploration


## 🔧 Installation

1. **Clone the Repository:**
   ```bash
   git clone git@github.com:adamjankelow/superlinked_demo.git
   cd superlinked_demo
   ```

2. **Set Up a Virtual Environment:**
   - Create a virtual environment:
     ```bash
     python3 -m venv venv
     ```
   - Activate the virtual environment (on Linux):
     ```bash
     source venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip build
   pip install --editable .         # or “pip install .” for a non-editable install
   ```

4. **Run the Streamlit App:**
   - From the root of the project directory, execute:
     ```bash
     streamlit run src/frontend/main.py
     ```

## Optional: Generate UMAP Vectors

If the UMAP vectors need to be created or updated, you can run the script to generate and save them. This is necessary if there have been changes to the food database or the UMAP configuration.

To generate the UMAP vectors, execute the following command from the root directory:

   ```bash
     python -m scripts.build_umap 
   ```
## Code Structure

### `src/frontend/main.py`
This module serves as the main **Streamlit application UI**. It manages user interactions, renders forms, and displays search results along with UMAP visualizations. To launch the app, use the command:

### `src/backend/search/queries.py`  
Core search logic using Superlinked.

### `src/backend/search/types.py`  
Data classes for shared context and parameters.

### `src/backend/ingest/loader.py`  
Utility functions for loading data and building the Superlinked app/index.

### `src/backend/ingest/schema.py`  
Defines the schema for food items in the database.

### `src/backend/features/umap.py`  
Helpers for UMAP projection and visualization; loads or caches `data/umap_df.parquet`.

### `scripts/build_umap.py`  
CLI script to generate and cache UMAP vectors for all embeddings, writing to `data/umap_df.parquet`.

### `data/`  
Directory for raw and derived data files (e.g., `sampled_food_db.parquet`, `umap_df.parquet`).

### `notebooks/`  
Jupyter notebooks for exploratory analysis and prototyping (e.g., multi-language search demo).

