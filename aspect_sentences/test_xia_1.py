import spacy

# 样本句子
sentence = "Finally, the delicious food and service was really great. The delicious food and service was not great."

# 加载英语语言模型
nlp = spacy.load("en_core_web_sm")

# 处理句子
doc = nlp(sentence)

# 打印每个单词及其依存关系
for token in doc:
    print(f"Word: {token.text}, POS: {token.pos_}, Dependency: {token.dep_}, Head: {token.head.text}")
