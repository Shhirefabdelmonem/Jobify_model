from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI()

# Globals
model = None
df = None
job_embeddings = None

# Define input schema
class UserProfile(BaseModel):
    name: str
    degree: str
    major: str
    gpa: float
    experience: int
    skills: str

# Load data and model on startup
@app.on_event("startup")
def load_model_and_data():
    global model, df, job_embeddings

    DATA_PATH = "data/wuzzuf_02_4_part3.csv"
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.rename(columns={
        'Job_Title': 'Job_Title',
        'Company_Name': 'Company_Name',
        'Job_Link': 'apply_link',
        'Skills': 'Skills',
        'Job_Description': 'Job_Description'
    }, inplace=True)
    df.dropna(subset=['Job_Title', 'Company_Name', 'Skills', 'Job_Description'], inplace=True)
    df['full_text'] = df['Job_Title'] + ' at ' + df['Company_Name'] + '. ' + \
                      df['Job_Description'] + ' Skills: ' + df['Skills']

    model = SentenceTransformer('all-MiniLM-L6-v2')
    job_embeddings = model.encode(df['full_text'].tolist(), show_progress_bar=True)


# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Job Recommendation API"}

# Recommendation endpoint
@app.post("/recommend")
def recommend_jobs(profile: UserProfile):
    try:
        global df, job_embeddings, model

        user_profile_text = f"{profile.degree} in {profile.major}, GPA {profile.gpa}, " \
                            f"{profile.experience} years experience. Skills: {profile.skills}"

        user_embedding = model.encode([user_profile_text])[0]
        similarities = cosine_similarity([user_embedding], job_embeddings)[0]
        df['similarity'] = similarities

        top_10 = df.sort_values(by='similarity', ascending=False).head(10)

        results = [
            {
                "job_title": row['Job_Title'],
                "company_name": row['Company_Name'],
                "skills": row['Skills'],
                "apply_link": row['apply_link'],
                "similarity": float(row['similarity'])
            }
            for _, row in top_10.iterrows()
        ]

        return {"recommended_jobs": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Run the app safely on Windows
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
