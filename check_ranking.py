import sys

sys.path.insert(0, ".")
sys.path.insert(0, "app")
import backend
import polars as pl

# Cargar parquet fresco (con fix de log-space)
df = pl.read_parquet("data/output/forecast.parquet")
print("rows:", df.height, "unique_ids:", df["unique_id"].n_unique())

# Ranking de tiendas sin SKU fijo (axis="store", fixed_peer=None)
# para sección 1
tabla_tiendas = backend.ranking_table(
    df,
    seccion="1",
    axis="store",
    fixed_peer=None,
    exclude=None,
    desc_map={},
    unidad="Unidades",
)
print("\n=== Ranking Tiendas sección 1 (sin SKU fijo) ===")
print("height:", tabla_tiendas.height)
if tabla_tiendas.height > 0:
    print("first rows:")
    print(tabla_tiendas.head(5))
    # wmape debe ser str porcentaje
    print("wMAPE type:", type(tabla_tiendas["wMAPE (%)"][0]))
    # unique_id debe existir
    print("has unique_id:", "unique_id" in tabla_tiendas.columns)
else:
    print("TABLE VACÍA - problema")

# Ranking SKU sin tienda fija (axis="sku", fixed_peer=None)
tabla_skus = backend.ranking_table(
    df,
    seccion="1",
    axis="sku",
    fixed_peer=None,
    exclude=None,
    desc_map={},
    unidad="Unidades",
)
print("\n=== Ranking SKU sección 1 (sin tienda fija) ===")
print("height:", tabla_skus.height)
if tabla_skus.height > 0:
    print("first rows:")
    print(tabla_skus.head(5))
    print("wMAPE type:", type(tabla_skus["wMAPE (%)"][0]))
    print("has unique_id:", "unique_id" in tabla_skus.columns)
else:
    print("TABLE VACÍA - problema")

# Ranking Tienda+SKU con SKU fijo (usando primer sku que aparezca en la tabla SKU)
if tabla_skus.height > 0:
    first_sku_code = tabla_skus["Código"][0]
    print(f"\n=== Ranking Tienda+SKU sección 1 con SKU fijo {first_sku_code} ===")
    tabla_ts = backend.ranking_table(
        df,
        seccion="1",
        axis="store",
        fixed_peer=first_sku_code,
        exclude=None,
        desc_map={},
        unidad="Unidades",
    )
    print("height:", tabla_ts.height)
    if tabla_ts.height > 0:
        print("first rows:")
        print(tabla_ts.head(5))
        print("wMAPE type:", type(tabla_ts["wMAPE (%)"][0]))
    else:
        print("TABLE VACÍA - problema")

print("\n=== DONE ===")
