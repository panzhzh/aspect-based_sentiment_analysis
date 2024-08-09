from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import sent_tokenize, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from aspect_keywords import aspect_keywords

# 读取数据
texts = []
with open('data.txt', 'r') as f:
    for line in f:
        text = line.strip().split('\t')[0]
        texts.append(text)

# 向量化文本数据
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# LDA模型
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(X)


# 主题词
def get_topic_words(lda_model, vectorizer, n_top_words=10):
    words = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        topic_words = [words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics[topic_idx] = topic_words
    return topics


topics = get_topic_words(lda, vectorizer)


def extract_aspect_sentences(text, aspect_keywords, topics):
    aspect_sentences = {aspect: {'text': '', 'sentiment': 'neutral', 'score': 0.0} for aspect in aspect_keywords}
    analyzer = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(text)

    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for topic, topic_words in topics.items():
            if any(word in words for word in topic_words):
                for aspect, keywords in aspect_keywords.items():
                    if any(keyword in words for keyword in keywords):
                        score = analyzer.polarity_scores(sentence)['compound']
                        aspect_sentences[aspect]['text'] = sentence
                        aspect_sentences[aspect]['score'] = round(score, 3)
                        aspect_sentences[aspect]['sentiment'] = 'positive' if score > 0 else 'negative'
                        break
    return aspect_sentences


with open('data.txt', 'r') as f:
    data = f.readlines()

for line in data:
    text = line.strip().split('\t')[0]
    aspect_sentences = extract_aspect_sentences(text, aspect_keywords, topics)
    overall_score = sum(aspect_sentences[aspect]['score'] for aspect in aspect_keywords) / len(aspect_keywords)
    overall_sentiment = 'positive' if overall_score > 0 else 'negative'
    aspect_sentences['overall'] = {'sentiment': overall_sentiment, 'score': round(overall_score, 3)}
    print({text: aspect_sentences})
