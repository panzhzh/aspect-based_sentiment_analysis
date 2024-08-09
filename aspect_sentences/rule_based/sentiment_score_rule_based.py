import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from aspect_sentences.aspect_keywords.aspect_keywords import aspect_keywords  # 确保根据你的文件结构调整此导入

# 初始化NLP模型和情感分析器
nlp = spacy.load('en_core_web_sm')
sia = SentimentIntensityAnalyzer()

# 定义文本列表
texts = [
    "The dishes taste good, but the selection is limited, breakfast is not very diverse. The waiter attitude was not good, a bit cold. The ambiance is fine, but the noise levels were high and the space felt cramped.",
    {"food": ["good", "limited", "not very diverse"], "service": ["not good", "a bit cold"], "ambiance": ["fine", "cramped"]},
    "Food is good, limited, not very diverse. Service: service is not good, a bit cold. Ambiance: ambiance is fine, cramped."
]

# 处理文本并计算情感得分的函数
def analyze_text(text):
    if isinstance(text, dict):  # 检查是否为字典类型（JSON数据）
        aspect_sentiments = {}
        for aspect, phrases in text.items():
            sentence = ', '.join(phrases) + '.'  # 将关键词列表转换为句子
            score = sia.polarity_scores(sentence)['compound']
            aspect_sentiments[aspect] = round(score, 2)
    else:
        doc = nlp(text)
        aspect_sentiments = {aspect: [] for aspect in aspect_keywords}  # 为每个文本重置
        # 分析句子结构并提取情感
        for sentence in doc.sents:
            for token in sentence:
                for aspect, keywords in aspect_keywords.items():
                    if token.lemma_.lower() in keywords:
                        # 发现相关方面，分析整个句子
                        score = sia.polarity_scores(sentence.text)['compound']
                        aspect_sentiments[aspect].append(score)
                        break  # 每个方面每个句子只分析一次
        # 计算每个方面的平均情感
        aspect_sentiments = {aspect: round(sum(scores) / len(scores), 2) if scores else 0 for aspect, scores in aspect_sentiments.items()}
    return aspect_sentiments

# 处理每段文本并打印结果
for text in texts:
    sentiments = analyze_text(text)
    print("Sentiments for text:", sentiments)
