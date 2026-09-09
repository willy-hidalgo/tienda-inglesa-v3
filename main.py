"""Launcher principal del pipeline Tienda Inglesa.

Ejecute ``uv run python main.py`` para abrir la interfaz operativa completa
(ingestión, forecasts, optimización SES+RLS, artefactos, validaciones y dashboard).
"""
from app.main import main


if __name__ == "__main__":
    main()
