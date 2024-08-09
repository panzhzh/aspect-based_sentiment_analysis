import spacy

# 加载 spaCy 的英文模型
nlp = spacy.load("en_core_web_sm")


def identify_modifiers_for_noun(doc, noun):
    description_phrases = []
    temp_pool = []

    # 查找名词及其直接修饰词
    for token in doc:
        if token.text == noun and token.pos_ == 'NOUN':
            for child in token.children:
                if child.dep_ in {'amod', 'det', 'advmod', 'neg', 'nummod', 'quantmod'} or child.pos_ == 'ADJ':
                    temp_pool.append(f"{child.text} ({child.dep_})")
                else:
                    if temp_pool:
                        description_phrases.append(' '.join(temp_pool) + f" modifies {noun}")
                        temp_pool = []
            # 检查名词的头部修饰词
            if token.head.dep_ in {'acomp', 'attr'}:
                temp_pool = []
                for child in token.head.children:
                    if child.dep_ in {'amod', 'advmod'}:
                        temp_pool.append(f"{child.text} ({child.dep_})")
                if temp_pool:
                    description_phrases.append(' '.join(temp_pool) + f" modifies {token.head.text}")

    # 查找间接修饰词
    for token in doc:
        if token.dep_ in {'amod', 'acomp'}:
            for child in token.children:
                if child.dep_ in {'advmod', 'amod'}:
                    description_phrases.append(f"{child.text} ({child.dep_}) modifies {token.text}")

    return description_phrases


# 示例用法
sentence = "The delicious food was really great."
doc = nlp(sentence)
noun = "food"
modifiers = identify_modifiers_for_noun(doc, noun)

print(f"'{noun}' 被修饰的描述词词组:")
for modifier in modifiers:
    print(modifier)
