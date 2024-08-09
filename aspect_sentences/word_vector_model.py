import gensim.downloader as api
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from aspect_keywords import aspect_keywords
import nltk

nltk.download('punkt')

# 加载预训练的词向量模型
model = api.load('glove-wiki-gigaword-100')

def expand_keywords(aspect_keywords, model, topn=10):
    expanded_keywords = {}
    for aspect, keywords in aspect_keywords.items():
        expanded_keywords[aspect] = set(keywords)
        for keyword in keywords:
            if keyword in model:
                similar_words = model.most_similar(keyword, topn=topn)
                for word, similarity in similar_words:
                    expanded_keywords[aspect].add(word)
    return expanded_keywords

expanded_aspect_keywords = expand_keywords(aspect_keywords, model)

def extract_aspect_sentences(text, aspect_keywords):
    sentences = sent_tokenize(text)
    aspect_sentences = {aspect: {'text': '', 'sentiment': 'neutral', 'score': 0.0} for aspect in aspect_keywords}
    analyzer = SentimentIntensityAnalyzer()

    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for aspect, keywords in aspect_keywords.items():
            for keyword in keywords:
                if keyword in words:
                    score = analyzer.polarity_scores(sentence)['compound']
                    aspect_sentences[aspect]['text'] = sentence
                    aspect_sentences[aspect]['score'] = round(score, 3)
                    aspect_sentences[aspect]['sentiment'] = 'positive' if score > 0 else 'negative'
                    break
    return aspect_sentences

with open('data.txt', 'r') as f:
    data = f.readlines()

for line in data:
    text = line.strip().split('\t')[0]  # 只获取文本部分
    aspect_sentences = extract_aspect_sentences(text, expanded_aspect_keywords)
    overall_score = sum(aspect_sentences[aspect]['score'] for aspect in aspect_keywords) / len(aspect_keywords)
    overall_sentiment = 'positive' if overall_score > 0 else 'negative'
    aspect_sentences['overall'] = {'sentiment': overall_sentiment, 'score': round(overall_score, 3)}
    print({text: aspect_sentences})
