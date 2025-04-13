import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class JobMatcher:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.df.dropna(subset=['Job_Title', 'Company_Name', 'Skills', 'Job_Description'], inplace=True)
        self.df['full_text'] = self.df['Job_Title'] + ' at ' + self.df['Company_Name'] + '. ' + \
                               self.df['Job_Description'] + ' Skills: ' + self.df['Skills']
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(self.df['full_text'].tolist(), show_progress_bar=True)

    def recommend(self, user_profile: str, top_k: int = 10):
        user_embedding = self.model.encode([user_profile])[0]
        similarities = cosine_similarity([user_embedding], self.embeddings)[0]
        self.df['similarity'] = similarities
        results = self.df.sort_values(by='similarity', ascending=False).head(top_k)

        return results[['Job_Title', 'Company_Name', 'Skills', 'apply_link', 'similarity']].to_dict(orient='records')
