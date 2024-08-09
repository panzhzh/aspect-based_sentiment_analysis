import spacy
import openai

# 设置你的OpenAI API Key
openai.api_key = 'YOUR_API_KEY'

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
        noun_counts[token.text] = noun_counts.get(token.text, 0) + 1

# 用于存储最终的修饰词和名词组合
noun_phrases = {}

# 用于追踪已处理的名词次数
processed_nouns = {}


def extract_modifiers(token):
    """提取修饰词"""
    modifiers = []
    for child in token.children:
        if child.dep_ in {'advmod', 'amod', 'det', 'neg', 'nummod', 'poss', 'prep', 'quantmod', 'acomp'}:
            modifiers.append(child.text)
            if child.dep_ == 'prep':
                for sub_child in child.children:
                    modifiers.append(sub_child.text)
    return modifiers


def extract_adj_modifiers(token):
    """提取形容词修饰词"""
    adj_modifiers = []
    for adj_child in token.children:
        if adj_child.dep_ in {'advmod', 'amod', 'det', 'neg', 'nummod', 'poss', 'prep', 'quantmod', 'acomp'}:
            adj_modifiers.append(adj_child.text)
            if adj_child.dep_ == 'prep':
                for adj_sub_child in adj_child.children:
                    adj_modifiers.append(adj_sub_child.text)
    for sibling in token.head.children:
        if sibling.dep_ == 'neg':
            adj_modifiers.append(sibling.text)
    return adj_modifiers


def gpt3_refine_modifiers(noun, context):
    """调用GPT-3来细化修饰词"""
    prompt = f"Extract the full phrase including modifiers for the noun '{noun}' in the following context: {context}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5
    )
    return response.choices[0].text.strip()


# 遍历文档中的每个词
for token in doc:
    if token.pos_ == 'NOUN':
        processed_nouns[token.text] = processed_nouns.get(token.text, 0)
        if processed_nouns[token.text] < noun_counts[token.text]:
            token_head = token.head.head if token.dep_ == 'conj' and token.head.pos_ == 'NOUN' else token.head
            modifiers = extract_modifiers(token)

            if token_head.pos_ == 'ADJ':
                adj_modifiers = extract_adj_modifiers(token_head)
                phrase = ' '.join(adj_modifiers + [token_head.text])
                if token.text in noun_phrases:
                    noun_phrases[token.text].append(phrase.strip())
                else:
                    noun_phrases[token.text] = [phrase.strip()]
            else:
                for sibling in token_head.children:
                    if sibling.pos_ == 'ADJ':
                        adj_modifiers = extract_adj_modifiers(sibling)
                        phrase = ' '.join(adj_modifiers + [sibling.text])
                        if token.text in noun_phrases:
                            noun_phrases[token.text].append(phrase.strip())
                        else:
                            noun_phrases[token.text] = [phrase.strip()]
                    elif sibling.dep_ == 'prep':
                        for sub_child in sibling.children:
                            modifiers.append(sub_child.text)

            # 在处理复杂依存关系时调用GPT-3
            context = sentence
            if len(modifiers) == 0:  # 如果初步提取没有找到修饰词，则调用GPT-3进行辅助
                gpt3_phrase = gpt3_refine_modifiers(token.text, context)
                noun_phrases[token.text] = [gpt3_phrase]
            else:
                noun_phrase = ' '.join(modifiers).strip()
                if token.text in noun_phrases:
                    noun_phrases[token.text].append(noun_phrase)
                else:
                    noun_phrases[token.text] = [noun_phrase]
            processed_nouns[token.text] += 1

# 结果后处理：去重、排序
for noun, phrases in noun_phrases.items():
    unique_phrases = list(set(phrase.strip().lower() for phrase in phrases if phrase.strip()))
    unique_phrases.sort()
    phrases_str = "', '".join(unique_phrases)
    print(f"{noun}: ['{phrases_str}']")
