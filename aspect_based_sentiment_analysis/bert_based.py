from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# 加载预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# 创建pipeline
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# 读取数据
data = []

with open('../aspect_sentences/dataset/data.txt', 'r') as f:
    for line in f:
        text, label = line.strip().split('\t')
        data.append(text)

# 预测
aspects = ['screen', 'battery', 'camera', 'design', 'performance', 'price', 'durability']

for text in data:
    aspect_sentiments = {}
    for aspect in aspects:
        aspect_text = f"The sentiment about {aspect} is {text}"
        result = nlp(aspect_text)
        aspect_sentiments[aspect] = result[0]['label']
    print(aspect_sentiments)
