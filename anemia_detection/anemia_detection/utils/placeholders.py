
"""
placeholders.py  —  REAL MODEL INFERENCE
════════════════════════════════════════════════════════════
SET YOUR MODEL PATHS HERE (only place you need to change):
════════════════════════════════════════════════════════════
"""

from PIL import Image
import numpy as np

# ╔══════════════════════════════════════════════════════════╗
# ║          ⚙️  SET YOUR MODEL PATHS HERE                  ║
# ╠══════════════════════════════════════════════════════════╣
# ║  Paste the full path to each model file.                ║
# ║                                                         ║
# ║  Windows examples:                                      ║
# ║    r"C:/Users/YourName/Anemia_Eye_Decation_CNN/best_anemia_cnn.pth"
# ║    r"C:/Users/YourName/Cinical/clinical_model.pkl"      ║
# ║                                                         ║
# ║  OR if you place model files next to app.py:            ║
# ║    "models/best_anemia_cnn.pth"                         ║
# ║    "models/clinical_model.pkl"                          ║
# ╚══════════════════════════════════════════════════════════╝

CNN_MODEL_PATH      = r"Anemia_Eye_Decation_CNN/best_anemia_cnn.pth"   # ← CHANGE THIS FILE PATH 
CLINICAL_MODEL_PATH = r"Cinical\clinical_model.pkl"                     # ← CHANGE THIS FILE PATH


# ══════════════════════════════════════════════════════════
#  Lazy model cache  (loads once, reuses on every prediction)
# ══════════════════════════════════════════════════════════
_cnn_model       = None
_clinical_bundle = None


def _load_cnn():
    global _cnn_model
    if _cnn_model is not None:
        return _cnn_model

    import torch
    import torch.nn as nn

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch, dropout=0.25):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(p=dropout),
            )
        def forward(self, x): return self.block(x)

    class AnemiaCNN(nn.Module):
        def __init__(self, num_classes=2, dropout=0.4):
            super().__init__()
            self.features = nn.Sequential(
                ConvBlock(3,   32,  0.20),
                ConvBlock(32,  64,  0.25),
                ConvBlock(64, 128,  0.25),
            )
            self.deep = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(256, 128), nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                nn.Dropout(p=dropout / 2),
                nn.Linear(128, num_classes),
            )
        def forward(self, x):
            x = self.features(x)
            x = self.deep(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = AnemiaCNN(num_classes=2, dropout=0.4)
    ckpt   = torch.load(CNN_MODEL_PATH, map_location=device)
    state  = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model  = model.to(device).eval()
    _cnn_model = (model, device)
    return _cnn_model


def _load_clinical():
    global _clinical_bundle
    if _clinical_bundle is not None:
        return _clinical_bundle
    import joblib
    _clinical_bundle = joblib.load(CLINICAL_MODEL_PATH)
    return _clinical_bundle


# ══════════════════════════════════════════════════════════
#  SECTION A — Clinical Model
# ══════════════════════════════════════════════════════════

def predict_clinical(clinical_inputs: dict) -> dict:
    try:
        import pandas as pd

        bundle    = _load_clinical()
        model     = bundle["model"]
        scaler    = bundle["scaler"]
        encoders  = bundle["encoders"]
        feat_cols = bundle["feature_cols"]

        raw = {
            "cell_diameter_um"                       : clinical_inputs.get("cell_diameter_um", 8.0),
            "nucleus_area_pct"                       : clinical_inputs.get("nucleus_area_pct", 0.0),
            "chromatin_density"                      : clinical_inputs.get("chromatin_density", 0.3),
            "cytoplasm_ratio"                        : clinical_inputs.get("cytoplasm_ratio", 1.0),
            "circularity"                            : clinical_inputs.get("circularity", 0.85),
            "eccentricity"                           : clinical_inputs.get("eccentricity", 0.2),
            "granularity_score"                      : clinical_inputs.get("granularity_score", 2.0),
            "lobularity_score"                       : clinical_inputs.get("lobularity_score", 1.0),
            "membrane_smoothness"                    : clinical_inputs.get("membrane_smoothness", 0.9),
            "cell_area_px"                           : clinical_inputs.get("cell_area_px", 200),
            "perimeter_px"                           : clinical_inputs.get("perimeter_px", 50),
            "mean_r"                                 : clinical_inputs.get("mean_r", 220),
            "mean_g"                                 : clinical_inputs.get("mean_g", 140),
            "mean_b"                                 : clinical_inputs.get("mean_b", 120),
            "stain_intensity"                        : clinical_inputs.get("stain_intensity", 0.55),
            "wbc_count_per_ul"                       : clinical_inputs.get("wbc_count_per_ul", 7000),
            "rbc_count_millions_per_ul"              : clinical_inputs.get("rbc_count_millions", 4.5),
            "hemoglobin_g_dl"                        : clinical_inputs.get("hemoglobin_g_dl", 12.0),
            "hematocrit_pct"                         : clinical_inputs.get("hematocrit_pct", 38.0),
            "platelet_count_per_ul"                  : clinical_inputs.get("platelet_count_per_ul", 250000),
            "mcv_fl"                                 : clinical_inputs.get("mcv_fl", 85.0),
            "mchc_g_dl"                              : clinical_inputs.get("mchc_g_dl", 33.0),
            "cytodiffusion_anomaly_score"            : 0.5,
            "cytodiffusion_classification_confidence": 0.5,
            "labeller_confidence_score"              : 0.8,
            "patient_age_group"                      : clinical_inputs.get("age_group", "Adult"),
            "patient_sex"                            : clinical_inputs.get("sex", "F"),
            "staining_protocol"                      : clinical_inputs.get("staining_protocol", "Wright"),
            "microscope_model"                       : clinical_inputs.get("microscope_model", "Zeiss_Axio"),
            "dataset_source"                         : clinical_inputs.get("dataset_source", "CytoData"),
        }
        df = pd.DataFrame([raw])

        # Feature engineering (mirrors clinical_model.py exactly)
        df["nucleus_to_cell_ratio"] = df["nucleus_area_pct"] / (df["cell_area_px"] + 1e-6)
        df["pallor_index"]          = (df["mean_r"] - df["mean_g"]) / (df["mean_r"] + df["mean_b"] + 1e-6)
        df["shape_complexity"]      = df["perimeter_px"] / (np.sqrt(df["cell_area_px"]) + 1e-6)
        df["mch_approx"]            = (df["hemoglobin_g_dl"] / df["rbc_count_millions_per_ul"].replace(0, np.nan)).fillna(0)
        df["anemia_risk_score"]     = (
            (df["hemoglobin_g_dl"] < 12).astype(int) +
            (df["mcv_fl"] < 80).astype(int) +
            (df["hematocrit_pct"] < 36).astype(int)
        )
        df["green_red_ratio"] = df["mean_g"] / (df["mean_r"] + 1e-6)
        df["morph_score"]     = (
            df["circularity"] * 0.3 +
            df["membrane_smoothness"] * 0.3 +
            (1 - df["eccentricity"]) * 0.2 +
            (1 - df["granularity_score"] / 10) * 0.2
        )

        for col, le in encoders.items():
            if col in df.columns:
                val = str(df[col].iloc[0])
                df[col] = le.transform([val])[0] if val in le.classes_ else 0

        X        = df[[c for c in feat_cols if c in df.columns]].fillna(0)
        X_scaled = scaler.transform(X)
        pred     = int(model.predict(X_scaled)[0])
        proba    = model.predict_proba(X_scaled)[0]
        prediction = "Anemic" if pred == 1 else "Non-Anemic"

        return {
            "prediction"    : prediction,
            "confidence"    : float(proba[pred]),
            "prob_anemic"   : float(proba[1]),
            "prob_normal"   : float(proba[0]),
            "recommendation": _recommendation(prediction, float(proba[pred])),
            "error"         : None,
        }

    except Exception as e:
        return _fallback_clinical(clinical_inputs, str(e))


# ══════════════════════════════════════════════════════════
#  SECTION B — CNN Eye Image Model
# ══════════════════════════════════════════════════════════

def predict_eye_image(pil_image: Image.Image) -> dict:
    try:
        import torch
        import torch.nn.functional as F
        from torchvision import transforms

        val_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

        model, device = _load_cnn()
        img = pil_image.convert("RGB")
        inp = val_tf(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out  = model(inp)
            prob = F.softmax(out, dim=1)[0]
            pred = int(out.argmax(1).item())

        prediction = "Anemic" if pred == 1 else "Non-Anemic"
        conf       = float(prob[pred].item())

        return {
            "prediction"    : prediction,
            "confidence"    : round(conf, 4),
            "prob_anemic"   : round(float(prob[1].item()), 4),
            "prob_normal"   : round(float(prob[0].item()), 4),
            "recommendation": _recommendation(prediction, conf),
            "error"         : None,
        }

    except Exception as e:
        return _fallback_eye(pil_image, str(e))


# ══════════════════════════════════════════════════════════
#  Fallbacks
# ══════════════════════════════════════════════════════════

def _fallback_clinical(inputs, err):
    hgb      = inputs.get("hemoglobin_g_dl", 12)
    p_anemic = round(float(np.clip((12 - hgb) * 0.08 + 0.35, 0.05, 0.95)), 4) if hgb < 12 else 0.18
    p_normal = round(1 - p_anemic, 4)
    pred     = "Anemic" if p_anemic > 0.5 else "Non-Anemic"
    return {
        "prediction"    : pred,
        "confidence"    : p_anemic if pred == "Anemic" else p_normal,
        "prob_anemic"   : p_anemic,
        "prob_normal"   : p_normal,
        "recommendation": _recommendation(pred, p_anemic),
        "error"         : f"Model load error — check CLINICAL_MODEL_PATH in placeholders.py\n\nDetails: {err}",
    }

def _fallback_eye(img, err):
    arr      = np.array(img.convert("RGB").resize((224, 224)))
    p_anemic = round(float(np.clip((220 - arr[:,:,0].mean()) / 150, 0.05, 0.95)), 4)
    p_normal = round(1 - p_anemic, 4)
    pred     = "Anemic" if p_anemic > 0.5 else "Non-Anemic"
    return {
        "prediction"    : pred,
        "confidence"    : p_anemic if pred == "Anemic" else p_normal,
        "prob_anemic"   : p_anemic,
        "prob_normal"   : p_normal,
        "recommendation": _recommendation(pred, p_anemic),
        "error"         : f"Model load error — check CNN_MODEL_PATH in placeholders.py\n\nDetails: {err}",
    }

def _recommendation(prediction: str, confidence: float) -> str:
    if prediction == "Anemic":
        if confidence >= 0.85:
            return "High-confidence anemia detected. Please consult a haematologist immediately. A CBC and iron-studies blood test is recommended."
        return "Possible anemia detected. Visit a physician for a blood test to confirm. Monitor symptoms: fatigue, pallor, dizziness."
    else:
        if confidence >= 0.85:
            return "No anemia detected with high confidence. Maintain a balanced diet rich in iron, B12, and folate. Annual check-ups recommended."
        return "Low anemia risk, but confidence is moderate. Consider a routine CBC if you experience fatigue or weakness."