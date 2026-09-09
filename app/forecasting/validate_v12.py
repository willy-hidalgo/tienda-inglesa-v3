"""v12 alias for the structural validator kept at validate_v11 for compatibility."""
from app.forecasting.validate_v11 import main, validate

__all__ = ["main", "validate"]

if __name__ == "__main__":
    raise SystemExit(main())
