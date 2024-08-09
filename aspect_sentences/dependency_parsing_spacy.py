import spacy
from spacy import displacy

# 加载英语模型
nlp = spacy.load("en_core_web_sm")

# 句子
sentence = "We had a fantastic meal at Armando al Pantheon! The food was delicious, especially the slow-roasted duck. The service was friendly, English-speaking, and efficient. The ambiance was warm, inviting, and comfortable, making for an enjoyable evening."

# 解析句子
doc = nlp(sentence)

# 打印句法依存关系
for token in doc:
    print(f"{token.text:10} {token.dep_:10} {token.head.text:10} {token.head.pos_:10} {[child for child in token.children]}")
