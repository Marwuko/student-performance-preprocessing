from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def build_standard_scaler_pipeline(numeric_cols):
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols)
        ],
        remainder='passthrough'
    )
