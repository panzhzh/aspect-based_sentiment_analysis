import spacy
import torch
from transformers import BertTokenizer, BertModel
from torch import nn

# 加载SpaCy和BERT模型
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# 样本句子
sentence = "Finally, the delicious food and service was really great. The delicious food and service was not great."

# 处理句子
doc = nlp(sentence)

# 提取所有修饰词短语
modifier_phrases = []

for token in doc:
    if token.pos_ == 'NOUN':
        # 如果名词是通过连接词连接的，更新其head为连接词连接的名词的head
        token_head = token.head.head if token.dep_ == 'conj' and token.head.pos_ == 'NOUN' else token.head

        # 检查名词的父节点是否为形容词
        if token_head.pos_ == 'ADJ':
            adj_modifiers = []
            for adj_child in token_head.children:
                if adj_child.dep_ in {'advmod', 'amod', 'det', 'neg', 'nummod', 'poss', 'prep', 'quantmod', 'acomp'}:
                    adj_modifiers.append(adj_child.text)
                    if adj_child.dep_ == 'prep':
                        for adj_sub_child in adj_child.children:
                            adj_modifiers.append(adj_sub_child.text)
            for sibling in token_head.head.children:
                if sibling.dep_ == 'neg':
                    adj_modifiers.append(sibling.text)
            phrase = ' '.join(adj_modifiers + [token_head.text])
            modifier_phrases.append(phrase.strip())

        # 添加名词的子节点修饰词
        for child in token.children:
            if child.dep_ in {'advmod', 'amod', 'det', 'neg', 'nummod', 'poss', 'prep', 'quantmod', 'acomp'}:
                phrase = [child.text]
                if child.dep_ == 'prep':
                    for sub_child in child.children:
                        phrase.append(sub_child.text)
                modifier_phrases.append(' '.join(phrase))

        # 检查名词的兄弟节点是否为形容词
        for sibling in token_head.children:
            if sibling.pos_ == 'ADJ':
                adj_modifiers = []
                for adj_child in sibling.children:
                    if adj_child.dep_ in {'advmod', 'amod', 'det', 'neg', 'nummod', 'poss', 'prep', 'quantmod', 'acomp'}:
                        adj_modifiers.append(adj_child.text)
                        if adj_child.dep_ == 'prep':
                            for adj_sub_child in adj_child.children:
                                adj_modifiers.append(adj_sub_child.text)
                for sibling_sibling in sibling.head.children:
                    if sibling_sibling.dep_ == 'neg':
                        adj_modifiers.append(sibling_sibling.text)
                phrase = ' '.join(adj_modifiers + [sibling.text])
                modifier_phrases.append(phrase)

            elif sibling.dep_ == 'prep':
                for sub_child in sibling.children:
                    modifier_phrases.append(sub_child.text)

# 使用BERT嵌入进行情感分析
class SentimentClassifier(nn.Module):
    def __init__(self, bert_model, hidden_size=768, num_classes=2):
        super(SentimentClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs[1]
        logits = self.classifier(cls_output)
        return logits

# 初始化情感分类器
model = SentimentClassifier(bert_model)
model.eval()

def predict_sentiment(phrase):
    inputs = tokenizer(phrase, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    return probabilities[0][1].item() - probabilities[0][0].item()

# 进行情感分析
sentiments = []
for phrase in modifier_phrases:
    sentiment = predict_sentiment(phrase)
    sentiments.append((phrase, sentiment))
    print(f"Phrase: {phrase}, Sentiment: {sentiment}")
