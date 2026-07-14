"""
Model registry — resolves and caches the right model instance for a scope.

Trained model filenames: BU_{tenant_id}_{bu_id}_{target}.joblib
                         Tenant_{tenant_id}_{target}.joblib
                         Market_{market_id}_{target}.joblib
                         Industry_{industry_id}_{target}.joblib

Fallback chain: BU → Tenant → Market → Industry

The scope_id passed here must match what train_all.py used:
  BU scope       → scope_id = "{tenant_id}_{bu_id}"
  Tenant scope   → scope_id = "{tenant_id}"
  Market scope   → scope_id = "{market_id}"
  Industry scope → scope_id = "{industry_id}"
"""
from typing import Dict
from ml.models.base_model import BaseMLModel
from ml.models.bu_model import BUModel
from ml.models.tenant_model import TenantModel
from ml.models.market_model import MarketModel
from ml.models.industry_model import IndustryModel

_MODEL_CACHE: Dict[str, BaseMLModel] = {}


def get_model(scope_level: str, scope_id: str) -> BaseMLModel:
    cache_key = f"{scope_level}:{scope_id}"
    if cache_key not in _MODEL_CACHE:
        if scope_level == "BU":
            _MODEL_CACHE[cache_key] = BUModel(scope_id=scope_id)
        elif scope_level == "Tenant":
            _MODEL_CACHE[cache_key] = TenantModel(scope_id=scope_id)
        elif scope_level == "Market":
            _MODEL_CACHE[cache_key] = MarketModel(scope_id=scope_id)
        else:
            _MODEL_CACHE[cache_key] = IndustryModel(scope_id=scope_id)
    return _MODEL_CACHE[cache_key]


def list_models() -> Dict[str, str]:
    return {k: v.model_version for k, v in _MODEL_CACHE.items()}


def invalidate(scope_level: str = None, scope_id: str = None):
    """Force reload after retraining."""
    global _MODEL_CACHE
    if scope_level and scope_id:
        _MODEL_CACHE.pop(f"{scope_level}:{scope_id}", None)
    else:
        _MODEL_CACHE.clear()
