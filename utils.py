import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import spacy


def get_ngram_counts(
    df: pd.DataFrame,
    text_col: str = "lemmas",
    ngram_range: tuple = (2, 3),
    min_size: int = 5,
):
    """get ngram counts from a dataframe"""
    try:
        vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_size)
        X = vectorizer.fit_transform(df[text_col])

        ngram_counts = pd.DataFrame(
            X.sum(axis=0).T, index=vectorizer.get_feature_names_out(), columns=["count"]
        )
    except Exception:
        ngram_counts = pd.DataFrame()
    return ngram_counts


def lemma_backwards_search(df, phrase, text_col: str = "lemmas"):
    """search for original phrase in a dataframe by keywords"""
    results = df[df[text_col].str.contains(phrase, na=False)]
    print(f"found {len(results)} reviews for the phrase '{phrase}'")
    return results


def lemmatize(nlp, text):
    """lemmatize a single text"""
    if pd.isna(text):
        return ""
    doc = nlp(str(text).lower())

    lemmas = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

    return " ".join(lemmas)


def lemmatize_pipe(texts):
    """batch lemmatization"""
    nlp = spacy.load("pl_core_news_lg")

    lemmas_list = []

    for doc in nlp.pipe(texts, batch_size=1000, disable=["ner", "parser"]):
        lemmas = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        lemmas_list.append(" ".join(lemmas))

    return lemmas_list


def review_prep(df: pd.DataFrame, score_range: tuple):
    """lemmatize a filtered dataframe"""

    filtered_df = df[df["score"].between(*score_range)].copy()

    texts = filtered_df["content"].fillna("").astype(str).tolist()

    result = filtered_df.assign(lemmas=lemmatize_pipe(texts))

    return result
