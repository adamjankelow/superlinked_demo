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
- Hard filtering by food categories
- Flexible combination of search criteria
- Optimized for intuitive exploration


## 🔧 Installation
Python version: >=3.11 recommended

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
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App:**
   - From the root of the project directory, execute:
     ```bash
     streamlit run code/semantic_food_search.py
     ```

## Code Structure

### `code/semantic_food_search.py`
Main Streamlit application entry point that handles the UI and coordinates between different search modes.
It should open in browser: http://localhost:8501

### `queries.py` 
Contains the core search functionality implementations:
- Simple semantic search
- Weighted multi-criteria search  
- Numeric range filtering
- Combined search with categories

### `utils.py`
Helper functions for:
- Data loading and preprocessing
- Building the Superlinked search index
- UMAP visualization
- Schema definitions

### `notebooks/`
The Jupyter notebook includes demonstrations of the following:
- Exploring potential improvements for enabling multi-language search with Streamlit.
- Translating a raw food database into a structured format using Large Language Models (LLMs).
- Utilizing additional parameters created for enhanced search capabilities with Superlinked.

