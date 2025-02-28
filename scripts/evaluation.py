import pandas as pd
from bert_score import score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_scores(cleaned_report, ai_generated_report):
    P, R, F1 = score([ai_generated_report], [cleaned_report], lang="en", verbose=True)
    bert_score_precision = P.mean().item()
    bert_score_recall = R.mean().item()
    bert_score_f1 = F1.mean().item()

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([cleaned_report, ai_generated_report])
    tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return (bert_score_precision, bert_score_recall, bert_score_f1, tfidf_score)
