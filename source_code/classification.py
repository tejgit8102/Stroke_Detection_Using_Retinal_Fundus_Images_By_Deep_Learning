import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import joblib

# Load classifier globally
clf = joblib.load("narrowing_detection_model.pkl")

def extract_vessel_features(binary_mask):
    if np.sum(binary_mask) == 0:
        return {'narrowing_ratio': 0, 'focal_narrowing_count': 0, 'max_width_drop': 0}

    skeleton = skeletonize(binary_mask.astype(bool))
    distance = distance_transform_edt(binary_mask)
    widths = distance[skeleton] * 2

    if len(widths) < 10:
        return {'narrowing_ratio': 0, 'focal_narrowing_count': 0, 'max_width_drop': 0}

    smooth = gaussian_filter1d(widths, sigma=2)
    drop = (np.max(smooth) - np.min(smooth)) / (np.max(smooth) + 1e-6)
    peaks, _ = find_peaks(-smooth, height=-np.percentile(smooth, 25))
    narrowing_ratio = (drop > 0.5) * 1

    return {
        'narrowing_ratio': narrowing_ratio,
        'focal_narrowing_count': len(peaks),
        'max_width_drop': drop
    }

def classify_narrowing(vessel_mask):
    features = extract_vessel_features(vessel_mask)
    features['vessel_density'] = np.sum(vessel_mask) / vessel_mask.size
    features['vessel_irregularity'] = np.std(distance_transform_edt(vessel_mask)) if np.sum(vessel_mask) > 0 else 0

    X = np.array([[
        features['narrowing_ratio'],
        features['max_width_drop'],
        features['focal_narrowing_count'],
        features['vessel_density'],
        features['vessel_irregularity']
    ]])

    pred = clf.predict(X)[0]
    prob = clf.predict_proba(X)[0][1]

    return pred, prob, features
