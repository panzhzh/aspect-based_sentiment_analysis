import spacy
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from aspect_sentences.aspect_keywords.aspect_keywords import aspect_keywords

# 加载spaCy的英语模型
nlp = spacy.load('en_core_web_sm')

def extract_aspect_phrases(text, aspect_keywords):
    aspect_phrases = {aspect: {'text': '', 'sentiment': 'neutral', 'score': 0.0} for aspect in aspect_keywords}
    analyzer = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(text)

    for sentence in sentences:
        doc = nlp(sentence)
        for token in doc:
            for aspect, keywords in aspect_keywords.items():
                if token.lemma_ in keywords:
                    phrase = ' '.join([child.text for child in token.children] + [token.text])
                    score = analyzer.polarity_scores(sentence)['compound']
                    aspect_phrases[aspect]['text'] = phrase
                    aspect_phrases[aspect]['score'] = round(score, 3)
                    aspect_phrases[aspect]['sentiment'] = 'positive' if score > 0 else 'negative'
                    break
    return aspect_phrases

with open('../dataset/data.txt', 'r') as f:
    data = f.readlines()

for line in data:
    text = line.strip().split('\t')[0]  # 只获取文本部分
    aspect_phrases = extract_aspect_phrases(text, aspect_keywords)
    overall_score = sum(aspect_phrases[aspect]['score'] for aspect in aspect_keywords) / len(aspect_keywords)
    overall_sentiment = 'positive' if overall_score > 0 else 'negative'
    aspect_phrases['overall'] = {'sentiment': overall_sentiment, 'score': round(overall_score, 3)}
    print({text: aspect_phrases})
