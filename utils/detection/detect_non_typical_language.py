import os
import numpy as np
import librosa
from pydub import AudioSegment, silence
from moviepy.editor import VideoFileClip

def detect_non_typical_language(
    video_path,
    repetition_threshold=0.90, # How similar audio segments must be to be considered a repetition
    monotony_threshold=0.2,    # How consistent speech/silence duration must be to be 'monotonous'
    min_silence_len_ms=300,    # Minimum silence between words to split them
    min_speech_len_ms=250      # Minimum length of a sound to be considered speech
):
    """
    Detects signs of non-typical language (echolalia, atypical prosody) from a video's audio.

    Args:
        video_path (str): Path to the video file.
        repetition_threshold (float): Similarity threshold for detecting echolalia.
        monotony_threshold (float): Standard deviation threshold for detecting monotony.
        min_silence_len_ms (int): Minimum duration of silence in milliseconds.
        min_speech_len_ms (int): Minimum duration of speech in milliseconds.

    Returns:
        dict: A dictionary like {'echolalia': 1, 'monotony': 0} indicating detection.
    """
    # --- 1. Extract Audio from Video ---
    try:
        print("Extracting audio from video...")
        video = VideoFileClip(video_path)
        audio_path = "temp_audio.wav"
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=16000)
        audio = AudioSegment.from_wav(audio_path)
    except Exception as e:
        print(f"Error processing video/audio: {e}. The video may not have an audio track.")
        return {'echolalia': 0, 'monotony': 0}

    # --- 2. Split Audio into Speech Segments ---
    # This separates spoken words/phrases from silence.
    print("Detecting speech segments...")
    speech_segments = silence.split_on_silence(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=audio.dBFS - 16, # Adjust threshold based on audio volume
        keep_silence=50
    )

    # Filter out segments that are too short to be meaningful speech
    speech_segments = [seg for seg in speech_segments if len(seg) >= min_speech_len_ms]

    if len(speech_segments) < 3:
        print("Not enough speech detected to perform analysis.")
        os.remove(audio_path)
        return {'echolalia': 0, 'monotony': 0}

    # --- 3. Analyze for Echolalia (Repetition) ---
    print("Analyzing for echolalia (repetition)...")
    echolalia_detected = False
    features = []
    for seg in speech_segments:
        # Convert audio segment to a numerical "fingerprint" (MFCC features)
        y = np.array(seg.get_array_of_samples(), dtype=np.float32)
        mfcc = librosa.feature.mfcc(y=y, sr=seg.frame_rate, n_mfcc=13)
        features.append(np.mean(mfcc.T, axis=0))

    # Compare each fingerprint to the others to find highly similar pairs
    repetition_count = 0
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            # Cosine similarity measures how similar two vectors are
            cos_sim = np.dot(features[i], features[j]) / (np.linalg.norm(features[i]) * np.linalg.norm(features[j]))
            if cos_sim > repetition_threshold:
                repetition_count += 1
    
    # If there are many repetitive pairs, flag for echolalia
    if repetition_count > len(speech_segments) / 2:
        echolalia_detected = True

    # --- 4. Analyze for Atypical Prosody (Monotony) ---
    print("Analyzing for atypical prosody (monotony)...")
    monotony_detected = False
    
    # Calculate the duration of each speech segment and the silence between them
    speech_durations = [len(seg) for seg in speech_segments]
    silence_durations = []
    non_silent_ranges = silence.detect_nonsilent(audio, min_silence_len=min_silence_len_ms, silence_thresh=audio.dBFS-16)
    if len(non_silent_ranges) > 1:
        for i in range(len(non_silent_ranges) - 1):
            silence_start = non_silent_ranges[i][1]
            silence_end = non_silent_ranges[i+1][0]
            silence_durations.append(silence_end - silence_start)

    # Monotony is indicated by unusually low variation in speech and silence lengths
    # We check if the standard deviation is very small compared to the average.
    if len(speech_durations) > 2 and np.mean(speech_durations) > 0:
        speech_monotony = np.std(speech_durations) / np.mean(speech_durations)
        if speech_monotony < monotony_threshold:
            monotony_detected = True
    
    if not monotony_detected and len(silence_durations) > 2 and np.mean(silence_durations) > 0:
        silence_monotony = np.std(silence_durations) / np.mean(silence_durations)
        if silence_monotony < monotony_threshold:
            monotony_detected = True

    # --- 5. Final Cleanup and Decision ---
    os.remove(audio_path) # Clean up the temporary audio file
    
    results = {
        'echolalia': 1 if echolalia_detected else 0,
        'monotony': 1 if monotony_detected else 0
    }

    print("\n--- Detection Summary ---")
    print(f"Echolalia (Repetition) Detected: {'Yes' if results['echolalia'] else 'No'}")
    print(f"Atypical Prosody (Monotony) Detected:  {'Yes' if results['monotony'] else 'No'}")

    return results

# --- Example Usage ---
# result = detect_non_typical_language("your_video.mp4")
# print("\nFinal Result:", result)