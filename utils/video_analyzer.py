import os

# --- 1. Import all detection functions from their respective files ---
# Note: Ensure the function names and filenames match your project exactly.
from utils.detection.aggressive_behaviour import detect_aggressive_behavior_improved
from utils.detection.detect_hyper_or_hypo import detect_hyper_or_hypo_reactivity
from utils.detection.detect_non_responsiveness import detect_non_responsiveness
from utils.detection.detect_non_typical_language import detect_non_typical_language
from utils.detection.detect_object_lining_up import detect_object_lining_up
from utils.detection.upper_limb_stereotypes import detect_upper_limb_stereotypies

# Assuming these files/functions exist as per your project structure
from utils.detection.eye_contact import detect_eye_contact
from utils.detection.detect_self_spinning import analyze_video_for_spinning
from utils.detection.self_hitting import SelfHitDetector

def analyze_video_for_traits(video_file_path):
    """
    Analyzes a video file by running a suite of behavior detection functions.

    Args:
        video_file_path (str): The full path to the video file.

    Returns:
        dict: A dictionary containing the detection results for each trait.
    """
    if not os.path.exists(video_file_path):
        print(f"Error: Video file not found at '{video_file_path}'")
        return None
    
    print(f"\n--- Starting Analysis for: {os.path.basename(video_file_path)} ---")
    
    # --- 2. Call each detection function with the video path ---
    # Each function will process the video and return its result (0 or 1, or a dict).
    
    print("\n[1/9] Checking for Aggressive Behavior...")
    aggressive_score = detect_aggressive_behavior_improved(video_file_path)

    print("\n[2/9] Checking for Hyper- or Hypo-reactivity...")
    reactivity_results = detect_hyper_or_hypo_reactivity(video_file_path)
    # The trait is positive if EITHER hyper OR hypo is detected.
    reactivity_score = max(reactivity_results.get('hyper', 0), reactivity_results.get('hypo', 0))

    print("\n[3/9] Checking for Non-Responsiveness...")
    non_responsiveness_score = detect_non_responsiveness(video_file_path)
    
    print("\n[4/9] Checking for Non-Typical Language...")
    language_results = detect_non_typical_language(video_file_path)
    # The trait is positive if EITHER echolalia OR monotony is detected.
    language_score = max(language_results.get('echolalia', 0), language_results.get('monotony', 0))

    print("\n[5/9] Checking for Object Lining-Up...")
    lining_up_score = detect_object_lining_up(video_file_path)

    print("\n[6/9] Checking for Upper Limb Stereotypies...")
    stereotypies_score = detect_upper_limb_stereotypies(video_file_path)

    # --- Placeholders for functions you will complete ---
    print("\n[7/9] Checking for Eye Contact...")
    eye_contact_score = detect_eye_contact(video_file_path) # Assuming this function exists

    print("\n[8/9] Checking for Self-Spinning...")
    self_spinning_score = analyze_video_for_spinning(video_file_path) # Assuming this function exists

    print("\n[9/9] Checking for Self-Hitting...")
    detector = SelfHitDetector()
    self_hitting_score = detector.process_video(video_file_path) # Assuming this function exists


    # --- 3. Compile the final results into a dictionary ---
    trait_scores = {
        'Absence or Avoidance of Eye Contact': eye_contact_score,
        'Aggressive Behavior': aggressive_score,
        'Hyper- or Hyporeactivity to Sensory Input': reactivity_score,
        'Non-Responsiveness to Verbal Interaction': non_responsiveness_score,
        'Non-Typical Language': language_score,
        'Object Lining-Up': lining_up_score,
        'Self-Hitting or Self-Injurious Behavior': self_hitting_score,
        'Self-Spinning or Spinning Objects': self_spinning_score,
        'Upper Limb Stereotypies': stereotypies_score,
    }
    
    print(f"\n--- Analysis Completed for: {os.path.basename(video_file_path)} ---")
    for trait, score in trait_scores.items():
        detection_status = "Detected" if score == 1 else "Not Detected"
        print(f"- {trait}: {detection_status}")
        
    return trait_scores