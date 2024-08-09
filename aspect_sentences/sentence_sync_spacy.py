import spacy

# 样本句子
sentence = "Finally, the not delicious food and service was really great. The delicious food and service was not great."

# 加载英语语言模型
nlp = spacy.load("en_core_web_sm")

# 处理句子
doc = nlp(sentence)

# 用于存储最终的修饰词和名词组合
noun_phrases = {}

# 遍历文档中的每个词
for token in doc:
    if token.pos_ == 'NOUN':
        # 找到每个名词和它的修饰词
        modifiers = []

        # 添加名词的子节点修饰词
        for child in token.children:
            if child.dep_ in {'advmod', 'amod', 'det', 'neg', 'nummod', 'poss', 'prep', 'quantmod', 'acomp'}:
                if child.dep_ == 'prep':
                    for sub_child in child.children:
                        modifiers.append(sub_child.text)
                if child.pos_ == 'ADJ':  # 如果修饰词是形容词，抽取形容词的修饰词
                    adj_modifiers = []
                    for adj_child in child.children:
                        if adj_child.dep_ in {'advmod', 'amod', 'det', 'neg', 'nummod', 'poss', 'prep', 'quantmod', 'acomp'}:
                            if adj_child.dep_ == 'prep':
                                for adj_sub_child in adj_child.children:
                                    adj_modifiers.append(adj_sub_child.text)
                            adj_modifiers.append(adj_child.text)
                    for sibling in child.head.children:
                        if sibling.dep_ == 'neg':
                            adj_modifiers.append(sibling.text)
                    modifiers += adj_modifiers
                modifiers.append(child.text)

        # 检查名词的父节点是否为形容词
        if token.head.pos_ == 'ADJ':
            adj_modifiers = []
            for adj_child in token.head.children:
                if adj_child.dep_ in {'advmod', 'amod', 'det', 'neg', 'nummod', 'poss', 'prep', 'quantmod', 'acomp'}:
                    if adj_child.dep_ == 'prep':
                        for adj_sub_child in adj_child.children:
                            adj_modifiers.append(adj_sub_child.text)
                    adj_modifiers.append(adj_child.text)
            for sibling in token.head.head.children:
                if sibling.dep_ == 'neg':
                    adj_modifiers.append(sibling.text)
            # 将形容词修饰词放在名词修饰词的前面
            modifiers = modifiers + [token.head.text] + adj_modifiers + [token.text]


        # 父节点不是形容词
        else:
            for sibling in token.head.children:
                if sibling.pos_ == 'ADJ':
                    adj_modifiers = []
                    for adj_child in sibling.children:
                        if adj_child.dep_ in {'advmod', 'amod', 'det', 'neg', 'nummod', 'poss', 'prep', 'quantmod', 'acomp'}:
                            if adj_child.dep_ == 'prep':
                                for adj_sub_child in adj_child.children:
                                    adj_modifiers.append(adj_sub_child.text)
                            adj_modifiers.append(adj_child.text)
                    for sibling_sibling in sibling.head.children:
                        if sibling_sibling.dep_ == 'neg':
                            adj_modifiers.append(sibling_sibling.text)
                    # 将形容词修饰词放在名词修饰词的前面
                    modifiers = modifiers + [token.text] + adj_modifiers + [sibling.text]
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

# 去除重复信息
for noun in noun_phrases:
    noun_phrases[noun] = list(set(noun_phrases[noun]))

# 输出所有的名词短语，以词组的形式展示并用逗号隔开
for noun, phrases in noun_phrases.items():
    phrases = [phrase.strip() for phrase in phrases if phrase]  # 去掉空白的短语并移除两端空格
    phrases_str = "', '".join(phrases)  # 用逗号和引号分隔短语
    print(f"{noun}: ['{phrases_str}']")
