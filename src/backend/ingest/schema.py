# src/backend/data/schema.py
from superlinked import framework as sl

class FoodItem(sl.Schema):
    """
    Schema definition for a food item in the database.
    Fields:
      - fdc_id: primary key
      - description: text
      - food_category: text
      - calories: numeric
    """
    fdc_id: sl.IdField
    description: sl.String
    food_category: sl.String
    calories: sl.Float
