import os
from fastapi import FastAPI, HTTPException
import tempfile
import requests
from pydantic import BaseModel

# --- 1. Import your project's components ---
from utils.video_analyzer import analyze_video_for_traits
from utils.model_predictor import load_model, generate_prediction_and_report

# --- 2. Define the expected request data structure using Pydantic ---
# This provides automatic data validation. The API will only accept
# requests that have a 'video_path' field which is a string.
class VideoRequest(BaseModel):
    videoUrl: str

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
    videoUrl = request.videoUrl

    try:

        # --- Step 1: Download video into a temp file ---
        with tempfile.NamedTemporaryFile( delete = False, suffix = ".mp4") as tmp_file:
            response = requests.get(videoUrl, stream = True)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download video from URL: {videoUrl}"
                )
            
            for chunk in response.iter_content(chunk_size = 8192):
                if chunk: 
                    tmp_file.write(chunk)

            tmp_path = tmp_file.name
        
        print(f"Download video to: {tmp_path}")


        # --- Run your existing analysis pipeline ---
        print(f"Starting analysis for: {tmp_path}")
        video_features = analyze_video_for_traits(tmp_path)

        if video_features:
            # Generate the final report using the modified function
            report = generate_prediction_and_report(video_features, PREDICTOR_MODEL)
            print(f"Analysis complete for: {tmp_path}")
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
    
    finally: 
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"Deleted temporary file: {tmp_path}")


@app.get("/ping", tags=["Health"])
def ping():
    return {"status": "alive"}

