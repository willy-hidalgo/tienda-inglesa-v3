cambios:

1. LA LÓGICA UTILIZADA EN LA SECCIÓN NO DEBE MODIFICARSE
2. aplicar para las tiendas la misma lógica de rls puro que se utiliza para la sección. por el momento, no calcular los estimados a nivel tienda/sku.
   (no se ha generado los pronósticos para el nivel tienda. además debería poder verlos la tabla de ranking para tienda, así como su gráfico.)
3. para los niveles sku y sku-tienda, en el periodo in-sample, calcular dos columnas de pronóstico utilizando respectivamente los coeficientes finales de los modelos de sección y tienda correspondientes (serían dos columnas para el yhat y dos columnas para valuehat). para cada valor de sku, sku-tienda se elegirá como modelo a utilizar aquel que tenga el menor wMAPE y en caso de empate el BIAS más cercano a cero.
4. el pronóstico obtenido con el mejor modelo en cada caso será el que se utilice en el dashboard para el ranking y el gráfico
   (no entendiste, para el periodo in-sample de los niveles sku y sku-tienda, se aplican los coeficientes de los modelos RLS de sección y tienda, se selecciona el mejor según el criterio del wmape y bias y se obtienen pronósticos que se obtendría no el modelo seleccionado, en in-sample no vamos a usar SES para nada en absoluto)
5. 2026-08-13 23:40:59,785 | INFO | ⏱ 1: derivación SKU+tienda vectorizada: iniciando…
   memory allocation of 125274912 bytes failed
   note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
   2026-08-13 23:45:04,541 | ERROR | 'crear pronósticos (RLS, paralelizable)' terminó con código de salida 3221226505
