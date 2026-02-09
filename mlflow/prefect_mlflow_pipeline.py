from prefect import flow, task
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
import unicodedata
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# ---------- TASKS ----------
df=pd.read_csv(r"D:\Downloads\reviews_badminton\data.csv")

def clean_text(text):
    text = str(text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    text = re.sub(r'pricedjust', 'priced just', text)
    text = re.sub(r'pricejust', 'price just', text)
    text = text.replace("o. k.", "ok").replace("o.k.", "ok")
    text = re.sub(r'read more', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]

    return " ".join(words)

df['clean_text'] = df['Review text'].apply(clean_text)
df[['Review text', 'clean_text']].head()

from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

df['normalized_text'] = df['clean_text'].apply(lemmatize_text)

df[['clean_text', 'normalized_text']].head()


pd.set_option('display.max_colwidth', None)

df[['Review text','clean_text', 'normalized_text']].head()

def sentiment_label(rating):
    if rating >= 4:
        return 1
    elif rating <= 2:
        return 0
    else:
        return None

df['sentiment'] = df['Ratings'].apply(sentiment_label)


df = df.dropna(subset=['sentiment'])

tfidf = TfidfVectorizer(
    max_features=5000,      
    ngram_range=(1,2)       
)

X_tfidf = tfidf.fit_transform(df['normalized_text'])


y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

@task
def train_model(X_train, y_train):
    mlflow.set_experiment("Prefect_MLflow_Demo")

    with mlflow.start_run(run_name="Prefect_LogReg"):
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 500)

        return model


@task
def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    f1_train = f1_score(y_train, y_train_pred)
    f1_test = f1_score(y_test, y_test_pred)

    mlflow.log_metric("f1_train", f1_train)
    mlflow.log_metric("f1_test", f1_test)

    print("Train F1:", f1_train)
    print("Test F1:", f1_test)


# ---------- FLOW ----------

@flow(name="MLflow + Prefect Pipeline")
def ml_pipeline(X_train, X_test, y_train, y_test):
    model = train_model(X_train, y_train)
    evaluate_model(model, X_train, X_test, y_train, y_test)


# ---------- RUN ----------
if __name__ == "__main__":
    # Assume data already prepared
    ml_pipeline(X_train, X_test, y_train, y_test)
