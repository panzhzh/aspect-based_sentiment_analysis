import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from aspect_keywords import aspect_keywords

nltk.download('punkt')

# 下载并加载spaCy的英语模型
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')
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
def expand_keywords(aspect_keywords, nlp, topn=10):
    expanded_keywords = {}
    for aspect, keywords in aspect_keywords.items():
        expanded_keywords[aspect] = set(keywords)
        for keyword in keywords:
            token = nlp(keyword)
            similar_words = [token.lemma_] + [w.text for w in token.vocab if w.has_vector and w.is_lower == token.is_lower]
            expanded_keywords[aspect].update(similar_words[:topn])
    return expanded_keywords

expanded_aspect_keywords = expand_keywords(aspect_keywords, nlp)

# 提取方面描述
def extract_aspect_sentences(text, aspect_keywords):
    sentences = sent_tokenize(text)
    aspect_sentences = {aspect: {'text': '', 'sentiment': 'neutral', 'score': 0.0} for aspect in aspect_keywords}

    for sentence in sentences:
        words = word_tokenize(sentence.lower())

        # BERT模型
        result = bert_analyzer(sentence)[0]
        sentiment = result['label']
        score = result['score'] if sentiment == 'POSITIVE' else -result['score']

        # 依存句法分析
        doc = nlp(sentence)
        for token in doc:
            for aspect, keywords in aspect_keywords.items():
                if token.lemma_ in keywords:
                    phrase = ' '.join([child.text for child in token.children] + [token.text])
                    aspect_sentences[aspect]['text'] = phrase
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
