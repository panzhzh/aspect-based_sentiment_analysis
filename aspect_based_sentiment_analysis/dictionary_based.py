import nltk
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from aspect_sentences.aspect_keywords.aspect_keywords import aspect_keywords, weights

nltk.download('punkt')


def aspect_based_sentiment(text, aspect_keywords, weights):
    words = word_tokenize(text.lower())
    aspect_sentiments = {}
    analyzer = SentimentIntensityAnalyzer()

    total_weighted_score = 0
    for aspect, keywords in aspect_keywords.items():
        aspect_sentiments[aspect] = {'text': '', 'sentiment': 'neutral', 'score': 0.0}
        combined_context = []
        for keyword in keywords:
            if keyword in words:
                aspect_index = words.index(keyword)
                context_words = words[max(0, aspect_index - 5):min(len(words), aspect_index + 6)]
                combined_context.extend(context_words)

        if combined_context:
            combined_text = ' '.join(combined_context)
            score = analyzer.polarity_scores(combined_text)['compound']
            aspect_sentiments[aspect]['text'] = combined_text
            aspect_sentiments[aspect]['score'] = round(score, 3)
            if score > 0:
                aspect_sentiments[aspect]['sentiment'] = 'positive'
            elif score < 0:
                aspect_sentiments[aspect]['sentiment'] = 'negative'
            total_weighted_score += score * weights[aspect]

    # 计算整体评分
    overall_sentiment = 'positive' if total_weighted_score > 0 else 'negative'
    overall_score = round(total_weighted_score, 3)

    aspect_sentiments['overall'] = {'sentiment': overall_sentiment, 'score': overall_score}

    return aspect_sentiments


with open('../aspect_sentences/dataset/data.txt', 'r') as f:
    data = f.readlines()

for line in data:
    text = line.strip()
    print(aspect_based_sentiment(text, aspect_keywords, weights))
