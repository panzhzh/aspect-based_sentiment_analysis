import spacy

# 样本句子
sentence = "Finally, The delicious food and service was really great. The delicious food and service was not great."

# 加载英语语言模型
nlp = spacy.load("en_core_web_sm")

# 处理句子
doc = nlp(sentence)

# 提取名词
nouns = [token.text for token in doc if token.pos_ == 'NOUN']

# 提取描述词词组
description_phrases = []

# 初始化临时池子
temp_pool = []

# 处理每个标记
for token in doc:
    if token.dep_ in {'advmod', 'amod', 'det', 'neg', 'nummod', 'poss', 'prep', 'quantmod', 'acomp'} or token.pos_ in {'ADJ'}:
        temp_pool.append(token.text)
    else:
        if temp_pool:
            # 检查池子中是否包含形容词
            if any(t.pos_ == 'ADJ' for t in nlp(' '.join(temp_pool))):
                description_phrases.append(' '.join(temp_pool))
            temp_pool = []

# 如果最后一个词也是修饰词，清空池子
if temp_pool:
    if any(t.pos_ == 'ADJ' for t in nlp(' '.join(temp_pool))):
        description_phrases.append(' '.join(temp_pool))

# 打印提取的名词和描述词词组
print("名词:", nouns)
print("描述词词组:", description_phrases)
