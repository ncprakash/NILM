"""Run seq2point inference on a raw aggregate power array."""
import numpy as np
import torch
from scipy.ndimage import median_filter
from model import DilatedSeq2Point

WINDOW = 599
HALF   = WINDOW // 2
AGG_MEAN, AGG_STD = 522.0, 814.0
MAX_POWER     = {"kettle": 3998, "fridge": 300,  "washing_machine": 2500, "dish_washer": 2500}
ON_THRESHOLDS = {"kettle": 2000, "fridge": 50,   "washing_machine": 50,   "dish_washer": 100}

DEVICE = torch.device("cpu")


def load_model(path):
    model = DilatedSeq2Point(dropout_cnn=0.1, dropout_fc=0.3, n_heads=5)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model, aggregate_watts, app_name, batch_size=512):
    agg_norm = ((aggregate_watts - AGG_MEAN) / AGG_STD).astype(np.float32)
    padded = np.pad(agg_norm, (HALF, HALF), mode="constant", constant_values=0.0)
    T = len(agg_norm)
    preds = np.zeros(T, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            batch = np.stack([padded[i:i + WINDOW] for i in range(start, end)])
            x = torch.from_numpy(batch).unsqueeze(1)
            out = model(x).numpy()
            preds[start:end] = out

    pred_w = np.clip(preds * MAX_POWER[app_name], 0, None)
    pred_w = median_filter(pred_w, size=5)
    return pred_w


def compute_metrics(pred_w, true_w, app_name):
    from sklearn.metrics import f1_score, mean_absolute_error
    thr = ON_THRESHOLDS[app_name]
    pred_on = (pred_w > thr).astype(int)
    true_on = (true_w > thr).astype(int)
    mae  = mean_absolute_error(true_w, pred_w)
    rmse = float(np.sqrt(np.mean((pred_w - true_w) ** 2)))
    f1   = float(f1_score(true_on, pred_on, zero_division=0))
    total_t = true_w.sum()
    eda = float(max(0, 1 - abs(pred_w.sum() - total_t) / max(total_t, 1))) if total_t > 1 else 0.0
    return {"MAE": mae, "RMSE": rmse, "F1": f1, "EDA": eda}
