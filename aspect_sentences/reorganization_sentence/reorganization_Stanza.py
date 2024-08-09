import stanza
import spacy
from aspect_sentences.aspect_keywords.aspect_keywords import aspect_keywords  # Import the aspect_keywords dictionary from the external file

# 下载和加载英语模型
stanza.download('en')
nlp_stanza = stanza.Pipeline('en')

# 加载spacy的英语模型
nlp_spacy = spacy.load('en_core_web_sm')

# 示例评论句子
#sentence = "Food delicious, slow the service, good the service"
#sentence = "We had a fantastic meal at Armando al Pantheon! The food was delicious, especially the slow-roasted duck. The service was friendly, English-speaking, and efficient. The ambiance was warm, inviting, and comfortable, making for an enjoyable evening."

sentence = "The portions are small, but the taste is okay, especially with a variety of drinks to choose from. The service is slow, and sometimes the waiters forget our orders. The environment is acceptable, but the cleanliness is average, with some greasy tables."
# 处理句子
doc_stanza = nlp_stanza(sentence)
doc_spacy = nlp_spacy(sentence)

# 统计每个名词的出现次数
noun_counts = {}
for sent in doc_stanza.sentences:
    for word in sent.words:
        if word.upos == 'NOUN':
            noun_counts[word.text] = noun_counts.get(word.text, 0) + 1

# 用于存储最终的修饰词和名词组合
noun_phrases = {}

# 用于追踪已处理的名词次数
processed_nouns = {}

def contains_noun_or_adj(phrase):
    sub_doc = nlp_stanza(phrase)
    return any(word.upos in ['NOUN', 'ADJ'] for sent in sub_doc.sentences for word in sub_doc.sentences[0].words)

def get_children(word, sent):
    return [w for w in sent.words if w.head == word.id]

# 遍历文档中的每个词
for sent in doc_stanza.sentences:
    for word in sent.words:
        if word.upos == 'NOUN':
            processed_nouns[word.text] = processed_nouns.get(word.text, 0)
            if processed_nouns[word.text] <= noun_counts[word.text]:
                # 找到每个名词和它的修饰词
                modifiers = []

                # 获取该词的head（父节点）
                head = next((w for w in sent.words if w.id == word.head), None)

                # 添加名词的子节点修饰词
                children = get_children(word, sent)
                for child in children:
                    if child.deprel in {'advmod', 'amod', 'det', 'neg', 'nummod', 'poss', 'prep', 'quantmod', 'acomp'}:
                        modifiers.append(child.text)
                        if child.deprel == 'prep':
                            prep_children = get_children(child, sent)
                            for sub_child in prep_children:
                                modifiers.append(sub_child.text)

                # 检查名词的父节点是否为形容词
                if head and head.upos == 'ADJ':
                    adj_modifiers = []
                    head_children = get_children(head, sent)
                    for adj_child in head_children:
                        if adj_child.deprel in {'advmod', 'amod', 'det', 'neg', 'nummod', 'poss', 'prep', 'quantmod', 'acomp'}:
                            adj_modifiers.append(adj_child.text)
                            if adj_child.deprel == 'prep':
                                adj_prep_children = get_children(adj_child, sent)
                                for adj_sub_child in adj_prep_children:
                                    adj_modifiers.append(adj_sub_child.text)
                    head_siblings = [w for w in sent.words if w.head == head.head]
                    for sibling in head_siblings:
                        if sibling.deprel == 'neg':
                            adj_modifiers.append(sibling.text)
                    phrase = ' '.join(adj_modifiers + [head.text])
                    if word.text in noun_phrases:
                        noun_phrases[word.text].append(phrase.strip())
                    else:
                        noun_phrases[word.text] = [phrase.strip()]
                else:
                    head_children = get_children(head, sent) if head else []
                    for sibling in head_children:
                        if sibling.upos == 'ADJ':
                            adj_modifiers = []
                            sibling_children = get_children(sibling, sent)
                            for adj_child in sibling_children:
                                if adj_child.deprel in {'advmod', 'amod', 'det', 'neg', 'nummod', 'poss', 'prep', 'quantmod', 'acomp'}:
                                    adj_modifiers.append(adj_child.text)
                                    if adj_child.deprel == 'prep':
                                        adj_prep_children = get_children(adj_child, sent)
                                        for adj_sub_child in adj_prep_children:
                                            adj_modifiers.append(adj_sub_child.text)
                            sibling_siblings = [w for w in sent.words if w.head == sibling.head]
                            for sibling_sibling in sibling_siblings:
                                if sibling_sibling.deprel == 'neg':
                                    adj_modifiers.append(sibling_sibling.text)
                            phrase = ' '.join(adj_modifiers + [sibling.text])
                            if word.text in noun_phrases:
                                noun_phrases[word.text].append(phrase.strip())
                            else:
                                noun_phrases[word.text] = [phrase.strip()]

                        elif sibling.deprel == 'prep':
                            prep_children = get_children(sibling, sent)
                            for sub_child in prep_children:
                                modifiers.append(sub_child.text)

                # 检查名词的兄弟节点是否为形容词
                siblings = [w for w in sent.words if w.head == word.head and w.deprel == 'amod']
                for sibling in siblings:
                    phrase = sibling.text
                    if word.text in noun_phrases:
                        noun_phrases[word.text].append(phrase.strip())
                    else:
                        noun_phrases[word.text] = [phrase.strip()]

                # 特殊处理: 检查并列结构中的形容词
                if head and head.deprel == 'parataxis' and head.upos == 'ADJ':
                    phrase = head.text
                    if word.text in noun_phrases:
                        noun_phrases[word.text].append(phrase.strip())
                    else:
                        noun_phrases[word.text] = [phrase.strip()]

                # 将修饰词与名词组合并存储，不包括名词本身
                noun_phrase = ' '.join(modifiers).strip()
                if word.text in noun_phrases:
                    noun_phrases[word.text].append(noun_phrase)
                else:
                    noun_phrases[word.text] = [noun_phrase]
                processed_nouns[word.text] += 1

# 将 spaCy 结果整合到 Stanza 结果中以提高准确性
for token in doc_spacy:
    if token.pos_ == 'ADJ' and token.dep_ == 'amod':
        head_noun = token.head.text
        if head_noun in noun_phrases:
            noun_phrases[head_noun].append(token.text)
        else:
            noun_phrases[head_noun] = [token.text]

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

# 转换结果格式
# 将 noun_phrases 中的键和值都转换为小写
noun_phrases_lower = {key.lower(): [phrase.lower() for phrase in phrases] for key, phrases in noun_phrases.items()}
result_phrases = {}
for key, values in aspect_keywords.items():
    result_phrases[key] = []
    for value in values:
        if value.lower() in noun_phrases_lower:  # 这里使用转换后的小写进行比较
            unique_phrases = list(set(phrase.strip().lower() for phrase in noun_phrases_lower[value.lower()] if phrase.strip()))
            final_phrases = [phrase for phrase in unique_phrases if not any(phrase != other and phrase in other for other in unique_phrases)]
            final_phrases = [phrase for phrase in final_phrases if contains_noun_or_adj(phrase)]
            if final_phrases:
                result_phrases[key].extend(final_phrases)

# 输出所有的名词短语，以词组的形式展示并用逗号隔开
for topic, sentences in result_phrases.items():
    if sentences:
        unique_sentences = list(set(sentences))
        combined_sentences = ', '.join(unique_sentences).replace(' .', '')
        print(f"{topic}: {topic} is {combined_sentences}.")

# 增加部分：输出名词和配对的修饰词
print("\nNouns and their associated phrases:")
for noun, phrases in result_phrases.items():
    if phrases:
        print(f"{noun}: {phrases}")