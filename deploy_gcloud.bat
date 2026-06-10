@echo off
REM ── Google Cloud Run deployment script ───────────────────────────────────────
REM Prerequisites:
REM   1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
REM   2. Run: gcloud auth login
REM   3. Run: gcloud config set project YOUR_PROJECT_ID

SET APP_NAME=nilm-app
SET REGION=us-central1
SET MEMORY=1Gi
SET CPU=1

echo.
echo ── Deploying %APP_NAME% to Google Cloud Run (%REGION%) ──
echo.

REM Enable required APIs (only needed once)
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

REM Deploy from source (Cloud Build handles the Docker build automatically)
gcloud run deploy %APP_NAME% ^
    --source . ^
    --platform managed ^
    --region %REGION% ^
    --allow-unauthenticated ^
    --memory %MEMORY% ^
    --cpu %CPU% ^
    --max-instances 3 ^
    --port 8080

echo.
echo ── Done. Your app URL is printed above. ──
