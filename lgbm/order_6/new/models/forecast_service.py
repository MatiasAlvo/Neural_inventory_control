
import json
import numpy as np
import lightgbm as lgb

class ForecastService:
    \"\"\"Batch LGBM-quantile forecaster for cumulative demand.
    Load with metadata.json, then call predict_quantiles(h, X) to get a matrix of quantile preds.\"\"\"
    def __init__(self, models_meta_path: str):
        with open(models_meta_path, "r") as f:
            meta = json.load(f)
        self.meta = meta
        self.models = {}  # (h, alpha) -> booster
        for m in meta.get("models", []):
            if m.get("path", "").endswith(".txt"):
                booster = lgb.Booster(model_file=m["path"])
                self.models[(m["h"], float(m["alpha"]))] = booster

        # Load feature schema per horizon
        self.schemas = {}
        for h in self.meta["config"]["HORIZONS"]:
            try:
                with open(models_meta_path.replace("metadata.json", f"feature_schema_h{h}.json"), "r") as f:
                    self.schemas[h] = json.load(f)
            except FileNotFoundError:
                pass

    def predict_quantiles(self, h: int, X: np.ndarray, alphas=None):
        if alphas is None:
            alphas = list(self.meta["config"]["ALPHAS"])
        preds = {}
        for a in alphas:
            mdl = self.models.get((h, float(a)))
            if mdl is None:
                continue
            preds[a] = mdl.predict(X)
        alphas_sorted = sorted(preds.keys())
        if len(alphas_sorted) == 0:
            return None, []
        mat = np.stack([preds[a] for a in alphas_sorted], axis=1)
        return mat, alphas_sorted
