from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from aspect_sentences.aspect_keywords.aspect_keywords import aspect_keywords, weights
import nltk

nltk.download('punkt')

# 读取数据
data = []
labels = []

with open('../aspect_sentences/dataset/data.txt', 'r') as f:
    for line in f:
        text, label = line.strip().split('\t')
        data.append(text)
        labels.append(label)

# 训练模型
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
model = make_pipeline(CountVectorizer(), LinearSVC())
model.fit(X_train, y_train)


def extract_aspects(text):
    words = word_tokenize(text.lower())
    found_aspects = []
    for aspect, keywords in aspect_keywords.items():
        for keyword in keywords:
            if keyword in words:
                found_aspects.append((aspect, keyword))
                break
    return found_aspects


analyzer = SentimentIntensityAnalyzer()

for text in X_test:
    found_aspects = extract_aspects(text)
    aspect_scores = {}
    aspect_texts = {}
    total_weighted_score = 0
    for aspect, keyword in found_aspects:
        aspect_index = text.lower().split().index(keyword)
        context_words = text.split()[max(0, aspect_index - 5):min(len(text.split()), aspect_index + 6)]
        aspect_text = ' '.join(context_words)
        sentiment_score = analyzer.polarity_scores(aspect_text)['compound']
        aspect_scores[aspect] = sentiment_score
        aspect_texts[aspect] = aspect_text
        total_weighted_score += sentiment_score * weights[aspect]

    overall_sentiment = 'positive' if total_weighted_score > 0 else 'negative'
    overall_score = round(total_weighted_score, 3)

    print(f"Text: {text}")
    for aspect, score in aspect_scores.items():
        sentiment_label = 'positive' if score > 0 else 'negative'
        print(
            f"  Aspect: {aspect}, Text: {aspect_texts[aspect]}, Sentiment: {sentiment_label}, Score: {round(score, 3)}")
    print(f"  Overall Sentiment: {overall_sentiment}, Overall Score: {overall_score}")
