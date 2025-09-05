import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- 1. Import your project's components ---
from utils.video_analyzer import analyze_video_for_traits
from utils.model_predictor import load_model, generate_prediction_and_report

# --- 2. Define the expected request data structure using Pydantic ---
# This provides automatic data validation. The API will only accept
# requests that have a 'video_path' field which is a string.
class VideoRequest(BaseModel):
    video_path: str

# --- 3. Initialize the FastAPI application ---
app = FastAPI(
    title="Autism Trait Detection API",
    description="An API to analyze videos for behavioral traits and predict autism likelihood.",
    version="1.0.0"
)

# --- 4. Load the ML model once when the server starts ---
# This is highly efficient as the model stays loaded in memory.
print("--- Initializing Prediction Model (this may take a moment)... ---")
PREDICTOR_MODEL = load_model()
print("--- Model Initialized. API is ready to accept requests. ---")


@app.post("/analyze", tags=["Analysis"])
def analyze_video_endpoint(request: VideoRequest):
    """
    Receives a path to a video file, runs the full analysis pipeline,
    and returns a comprehensive report as a JSON object.
    """
    video_file_path = request.video_path

    # Validate that the file path provided by the Node.js server actually exists
    if not os.path.exists(video_file_path):
        # FastAPI handles this by sending a clean 404 Not Found error
        raise HTTPException(
            status_code=404, 
            detail=f"Video file not found at the provided path: {video_file_path}"
        )

    try:
        # --- Run your existing analysis pipeline ---
        print(f"Starting analysis for: {video_file_path}")
        video_features = analyze_video_for_traits(video_file_path)

        if video_features:
            # Generate the final report using the modified function
            report = generate_prediction_and_report(video_features, PREDICTOR_MODEL)
            print(f"Analysis complete for: {video_file_path}")
            # FastAPI automatically converts the dictionary to a JSON response
            return report
        else:
            raise HTTPException(
                status_code=500, 
                detail="Video analysis failed to return any features."
            )

    except Exception as e:
        # Catch any other unexpected errors during the process
        print(f"An unexpected error occurred during analysis: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"An internal server error occurred: {str(e)}"
        )