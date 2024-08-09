import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gensim.downloader as api
import spacy
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from aspect_keywords import aspect_keywords

nltk.download('punkt')

# 加载预训练的词向量模型
word_vector_model = api.load('glove-wiki-gigaword-100')

# 下载并加载spaCy的英语模型
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# 加载spaCy的英语模型
nlp = spacy.load('en_core_web_sm')

# 加载预训练的BERT模型
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
bert_analyzer = pipeline('sentiment-analysis', model=bert_model, tokenizer=bert_tokenizer)

# 读取数据
def read_data(file_path):
    with open(file_path, 'r') as f:
        return [line.strip().split('\t')[0] for line in f.readlines()]

# 扩展关键词列表
def expand_keywords(aspect_keywords, model, topn=10):
    expanded_keywords = {}
    for aspect, keywords in aspect_keywords.items():
        expanded_keywords[aspect] = set(keywords)
        for keyword in keywords:
            if keyword in model:
                similar_words = model.most_similar(keyword, topn=topn)
                for word, _ in similar_words:
                    expanded_keywords[aspect].add(word)
    return expanded_keywords

expanded_aspect_keywords = expand_keywords(aspect_keywords, word_vector_model)

# 提取方面描述
def extract_aspect_sentences(text, aspect_keywords):
    sentences = sent_tokenize(text)
    aspect_sentences = {aspect: {'text': '', 'sentiment': 'neutral', 'score': 0.0} for aspect in aspect_keywords}
    analyzer = SentimentIntensityAnalyzer()

    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        # 关键词匹配和词向量模型
        for aspect, keywords in aspect_keywords.items():
            if any(keyword in words for keyword in keywords):
                score = analyzer.polarity_scores(sentence)['compound']
                aspect_sentences[aspect]['text'] = sentence
                aspect_sentences[aspect]['score'] = round(score, 3)
                aspect_sentences[aspect]['sentiment'] = 'positive' if score > 0 else 'negative'

        # 依存句法分析
        doc = nlp(sentence)
        for token in doc:
            for aspect, keywords in aspect_keywords.items():
                if token.lemma_ in keywords:
                    phrase = ' '.join([child.text for child in token.children] + [token.text])
                    score = analyzer.polarity_scores(sentence)['compound']
                    aspect_sentences[aspect]['text'] = phrase
                    aspect_sentences[aspect]['score'] = round(score, 3)
                    aspect_sentences[aspect]['sentiment'] = 'positive' if score > 0 else 'negative'

        # BERT模型
        result = bert_analyzer(sentence)[0]
        sentiment = result['label']
        score = result['score'] if sentiment == 'POSITIVE' else -result['score']
        for aspect, keywords in aspect_keywords.items():
            if any(keyword in words for keyword in keywords):
                aspect_sentences[aspect]['text'] = sentence
                aspect_sentences[aspect]['score'] = round(score, 3)
                aspect_sentences[aspect]['sentiment'] = 'positive' if score > 0 else 'negative'
    return aspect_sentences

# 主函数
def main():
    data = read_data('data.txt')
    output = []
    for text in data:
        aspect_sentences = extract_aspect_sentences(text, expanded_aspect_keywords)
        overall_score = sum(aspect_sentences[aspect]['score'] for aspect in aspect_keywords) / len(aspect_keywords)
        overall_sentiment = 'positive' if overall_score > 0 else 'negative'
        aspect_sentences['overall'] = {'sentiment': overall_sentiment, 'score': round(overall_score, 3)}
        output.append({text: aspect_sentences})
    for result in output:
        print(result)

if __name__ == "__main__":
    main()
