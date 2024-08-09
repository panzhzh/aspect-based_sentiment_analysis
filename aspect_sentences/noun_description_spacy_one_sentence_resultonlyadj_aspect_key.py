import spacy

# 样本句子
#sentence = "Finally, the delicious food and service was really great. The delicious food and service was not great."
#sentence = "Good Filipino food! Tortang talong was a winner. Service is good and staff are all friendly. Will come back again to try out other dishes on the menu."
sentence ="Great place. Everything we had was delicious, very well prepared and nicely served. Highly recommend the ravioli, best we ever had. Service was excellent too. Price very reasonable."
# 加载英语语言模型
nlp = spacy.load("en_core_web_sm")

# 处理句子
doc = nlp(sentence)

# 统计每个名词的出现次数
noun_counts = {}
for token in doc:
    if token.pos_ == 'NOUN':
        noun_counts[token.text] = noun_counts.get(token.text, 0) + 1

# 用于存储最终的修饰词和名词组合
noun_phrases = {}

# 用于追踪已处理的名词次数
processed_nouns = {}

def contains_noun_or_adj(phrase):
    sub_doc = nlp(phrase)
    return any(token.pos_ in ['NOUN', 'ADJ'] for token in sub_doc)


# 遍历文档中的每个词
for token in doc:
    if token.pos_ == 'NOUN':
        processed_nouns[token.text] = processed_nouns.get(token.text, 0)
        if processed_nouns[token.text] <= noun_counts[token.text]:
            # 找到每个名词和它的修饰词
            modifiers = []

            # 如果名词是通过连接词连接的，更新其head为连接词连接的名词的head
            token_head = token.head.head if token.dep_ == 'conj' and token.head.pos_ == 'NOUN' else token.head

            # 添加名词的子节点修饰词
            for child in token.children:
                if child.dep_ in {'advmod', 'amod', 'det', 'neg', 'nummod', 'poss', 'prep', 'quantmod', 'acomp'}:
                    modifiers.append(child.text)
                    if child.dep_ == 'prep':
                        for sub_child in child.children:
                            modifiers.append(sub_child.text)

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
                if token.text in noun_phrases:
                    noun_phrases[token.text].append(phrase.strip())
                else:
                    noun_phrases[token.text] = [phrase.strip()]
            else:
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
                        if token.text in noun_phrases:
                           noun_phrases[token.text].append(phrase.strip())
                        else:
                           noun_phrases[token.text] = [phrase.strip()]

                    else:
                        if sibling.dep_ == 'prep':
                            for sub_child in sibling.children:
                                modifiers.append(sub_child.text)

            # 将修饰词与名词组合并存储，不包括名词本身
            noun_phrase = ' '.join(modifiers).strip()
            if token.text in noun_phrases:
                noun_phrases[token.text].append(noun_phrase)
            else:
                noun_phrases[token.text] = [noun_phrase]
            processed_nouns[token.text] += 1

# 输出所有的名词短语，以词组的形式展示并用逗号隔开
for noun, phrases in noun_phrases.items():
    unique_phrases = list(set(phrase.strip().lower() for phrase in phrases if phrase.strip()))
    # 移除被其他短语包含的短语
    final_phrases = [phrase for phrase in unique_phrases if not any(phrase != other and phrase in other for other in unique_phrases)]
    # 仅保留包含名词或形容词的短语
    final_phrases = [phrase for phrase in final_phrases if contains_noun_or_adj(phrase)]
    if final_phrases:  # 仅输出包含名词或形容词的结果
        phrases_str = "', '".join(final_phrases)  # 用逗号和引号分隔短语
        print(f"{noun}: ['{phrases_str}']")
