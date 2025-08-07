import os

VIDEO_FILE = os.path.join("data", "patient_video.mp4")

OUTPUT_FEATURES_CSV = "analyzed_video_features.csv"

MODEL_PATH = "autism_predictor_model.joblib"

TRAIT_COLUMNS = [
    'Absence or Avoidance of Eye Contact',
    'Aggressive Behavior',
    'Hyper- or Hyporeactivity to Sensory Input',
    'Non-Responsiveness to Verbal Interaction',
    'Non-Typical Language',
    'Object Lining-Up',
    'Self-Hitting or Self-Injurious Behavior',
    'Self-Spinning or Spinning Objects',
    'Upper Limb Stereotypies',
]