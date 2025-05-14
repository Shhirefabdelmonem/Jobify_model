# BERT Job Recommender API

This is a FastAPI-based job recommender system using a BERT model.

## Project Structure

```
.
├── data/
│   └── wuzzuf_02_4_part3.csv
├── model/
│   └── job_model.py
├── main.py
├── requirements.txt
├── Procfile
└── README.md
```

## Requirements
- Python 3.8+
- FastAPI
- Uvicorn
- Pydantic
- Pandas
- scikit-learn

## Local Development

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the server:
   ```bash
   uvicorn main:app --reload
   ```

## Deploying on Railway

1. Push your code to a GitHub repository.
2. Go to [Railway](https://railway.app/) and create a new project.
3. Connect your GitHub repo.
4. Ensure the following files are present:
   - `requirements.txt`
   - `Procfile`
   - `main.py`
   - `data/wuzzuf_02_4_part3.csv`
5. Railway will automatically detect the `Procfile` and install dependencies.
6. The app will be available at the generated Railway URL.

## API Endpoints

- `GET /` - Health check
- `POST /recommend` - Get job recommendations

## Example Request

```
POST /recommend
Content-Type: application/json
{
  "name": "John Doe",
  "degree": "BSc",
  "major": "Computer Science",
  "gpa": 3.5,
  "experience": 2,
  "skills": "Python, Machine Learning, Data Analysis"
}
``` 