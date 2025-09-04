import pandas as pd
from utils.model_predictor import train_and_save_model
# from video_analyzer import analyze_video_for_traits
# from model_predictor import load_model, generate_prediction_and_report
from config.config import VIDEO_FILE, OUTPUT_FEATURES_CSV, TRAIT_COLUMNS
from utils.video_analyzer import analyze_video_for_traits
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# def main():
#     """
#     Main function to orchestrate the entire video analysis and prediction pipeline.
#     """
#     # 1. Load the machine learning model
#     predictor_model = load_model()

#     # 2. Analyze the video to get the features
#     video_features = analyze_video_for_traits(VIDEO_FILE)

#     if video_features:
#         # 3. Save the extracted features to a CSV
#         video_id = os.path.basename(VIDEO_FILE)
#         features_df = pd.DataFrame([video_features], columns=TRAIT_COLUMNS)
#         features_df.insert(0, 'Video_ID', video_id)
#         features_df.to_csv(OUTPUT_FEATURES_CSV, index=False)
#         print(f"\nExtracted features saved to '{OUTPUT_FEATURES_CSV}'.")

#         # 4. Generate the prediction and report
#         generate_prediction_and_report(video_features, predictor_model)


# train_and_save_model()
VIDEO_FILE = "1.mp4"

if os.path.exists(VIDEO_FILE):
    analyze_video_for_traits(VIDEO_FILE)