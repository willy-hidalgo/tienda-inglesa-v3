# Datos de entrada locales

Este directorio contiene fuentes comerciales/proprietary y está excluido de
Git. Los archivos se deben aprovisionar por el canal autorizado antes de correr
la ingesta; sus nombres y rutas operativas están definidos en `settings.py`.

No agregar excepciones individuales a `.gitignore` para CSV, XLSX, DAT o
Parquet de negocio. La trazabilidad del código y del lockfile no sustituye el
control de acceso ni el versionado del dataset fuente.
