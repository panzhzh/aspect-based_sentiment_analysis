import spacy

# 样本句子
sentence = "Finally, the delicious food and service was really great. The delicious food and service was not great."

# 加载英语语言模型
nlp = spacy.load("en_core_web_sm")

# 处理句子
doc = nlp(sentence)

# 统计每个名词的出现次数
noun_counts = {}
for token in doc:
    if token.pos_ == 'NOUN':
        if token.text in noun_counts:
            noun_counts[token.text] += 1
        else:
            noun_counts[token.text] = 1

# 用于存储最终的修饰词和名词组合
noun_phrases = {}

# 用于追踪已处理的名词次数
processed_nouns = {}

# 遍历文档中的每个词
for token in doc:
    if token.pos_ == 'NOUN':
        # 初始化已处理的名词次数
        if token.text not in processed_nouns:
            processed_nouns[token.text] = 0

        # 检查是否已经达到该名词的最大次数
        if processed_nouns[token.text] < noun_counts[token.text]:
            # 找到每个名词和它的修饰词
            modifiers = []

            # 如果名词是通过连接词连接的，更新其head为连接词连接的名词的head
            if token.dep_ == 'conj' and token.head.pos_ == 'NOUN':
                token_head = token.head.head
            else:
                token_head = token.head

            # 添加名词的子节点修饰词
            for child in token.children:
                if child.dep_ in {'advmod', 'amod', 'det', 'neg', 'nummod', 'poss', 'prep', 'quantmod', 'acomp'}:
                    modifiers.append(child.text)
                    # 如果修饰词是介词，添加介词的子节点
                    if child.dep_ == 'prep':
                        for sub_child in child.children:
                            modifiers.append(sub_child.text)

            # 检查名词的父节点
            if token_head.pos_ == 'ADJ':
                adj_modifiers = []
                # 捕捉父节点(形容词)的修饰词
                for adj_child in token_head.children:
                    if adj_child.dep_ in {'advmod', 'amod', 'det', 'neg', 'nummod', 'poss', 'prep', 'quantmod', 'acomp'}:
                        adj_modifiers.append(adj_child.text)
                        if adj_child.dep_ == 'prep':
                            for adj_sub_child in adj_child.children:
                                adj_modifiers.append(adj_sub_child.text)
                # 检查形容词的兄弟节点是否为否定词（如 not）
                for sibling in token_head.head.children:
                    if sibling.dep_ == 'neg':
                        adj_modifiers.append(sibling.text)
                modifiers.extend(adj_modifiers)
                modifiers.append(token_head.text)
            else:
                # 遍历父节点的子节点
                for sibling in token_head.children:
                    if sibling.pos_ == 'ADJ':
                        adj_modifiers = []
                        # 捕捉这个形容词的修饰词
                        for adj_child in sibling.children:
                            if adj_child.dep_ in {'advmod', 'amod', 'det', 'neg', 'nummod', 'poss', 'prep', 'quantmod', 'acomp'}:
                                adj_modifiers.append(adj_child.text)
                                if adj_child.dep_ == 'prep':
                                    for adj_sub_child in adj_child.children:
                                        adj_modifiers.append(adj_sub_child.text)
                        # 检查形容词的兄弟节点是否为否定词（如 not）
                        for sibling_sibling in sibling.head.children:
                            if sibling_sibling.dep_ == 'neg':
                                adj_modifiers.append(sibling_sibling.text)
                        modifiers.extend(adj_modifiers)
                        modifiers.append(sibling.text)
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
            # 更新已处理的名词次数
            processed_nouns[token.text] += 1

# 输出所有的名词短语，以词组的形式展示并用逗号隔开
for noun, phrases in noun_phrases.items():
    phrases = [phrase.strip() for phrase in phrases if phrase]  # 去掉空白的短语并移除两端空格
    phrases_str = "', '".join(phrases)  # 用逗号和引号分隔短语
    print(f"{noun}: ['{phrases_str}']")
