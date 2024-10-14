import spacy
# 导入aspect_keywords
from aspect_sentences.aspect_keywords.aspect_keywords import aspect_keywords

# 现在可以使用aspect_keywords在您的代码中

# 加载英语模型
nlp = spacy.load("en_core_web_sm")

def find_adjectives_modifying_nouns(doc, keyword_dict):
    noun_adj_pairs = []
    for token in doc:
        if (token.pos_ == "NOUN" or token.pos_ == "PROPN") and token.text.lower() in keyword_dict:
            for potential_adj in doc:
                if potential_adj.pos_ == "ADJ" and is_modifying_noun(token, potential_adj):
                    noun_adj_pairs.append((token, potential_adj))
    return noun_adj_pairs

def is_modifying_noun(noun, adjective):
    # 直接检查形容词是否依存于名词
    if adjective.head == noun:
        return True

    # 通过依存路径寻找间接连接
    current_token = adjective
    path_to_noun = set()  # 记录路径上的所有token

    # 沿依存树向上追踪，直到达到根节点。这里是追踪形容词相关的依存路径。在这种情况下，需要追踪从形容词开始的依存路径，看是否可以通过一系列的头（head）关系到达目标名词。
    while current_token.head != current_token:
        current_token = current_token.head
        path_to_noun.add(current_token)

        # 如果在路径中找到了目标名词，则形容词间接修饰该名词
        if current_token == noun:
            return True

        # 如果当前token是助动词或连词，并且其父节点也在路径上，检查是否构成间接修饰关系
        if current_token.pos_ in ["AUX", "VERB", "SCONJ", "CCONJ"] and current_token.head in path_to_noun:
            siblings = list(current_token.children)
            # 检查所有与当前token同级的其他节点，看它们是否包括目标名词
            for sibling in siblings:
                if sibling == noun:
                    return True
                if sibling.pos_ == "NOUN" and sibling.dep_ in ["attr", "nsubj", "dobj"]:
                    # 如果有与目标名词语义相关的兄弟节点，也认为形容词修饰该名词
                    return True

    # 考虑名词后的形容词，例如 "The keys are blue."
    if adjective.dep_ == "acomp" and adjective.head.pos_ == "VERB":
        # 如果形容词是谓语动词的补足语，检查该动词是否与名词有关联
        action_subj = [child for child in adjective.head.children if child.dep_ in ["nsubj", "nsubjpass"]]
        if any(subj == noun for subj in action_subj):
            return True

    return False

def check_structure(doc, adj_token):
    structures = []
    adj_idx = adj_token.i

    # 向前检查
    front_tokens = []
    for i in range(adj_idx - 1, -1, -1):
        token = doc[i]
        if token.pos_ in {"NOUN", "PROPN", "AUX", "CCONJ"} or token.is_punct:
            break
        front_tokens.insert(0, token.text)
    if front_tokens:
        phrase = " ".join(front_tokens + [adj_token.text])
        structures.append(phrase)

    # 向后检查
    back_tokens = []
    for i in range(adj_idx + 1, len(doc)):
        token = doc[i]
        if token.pos_ in {"NOUN", "PROPN", "VERB", "AUX", "CCONJ"} or token.is_punct:
            break
        back_tokens.append(token.text)
    if front_tokens:
        if(back_tokens):
           phrase = " ".join(back_tokens)
           structures.append(phrase)
    else:
        phrase = " ".join([adj_token.text]+back_tokens)
        structures.append(phrase)
    return structures


def split_sentences(doc):
    # 初始句子列表
    sentences = []
    # 临时句子起始索引
    temp_sent_start = 0
    # 遍历文档中的每个token
    for i, token in enumerate(doc):
        # 检测到连词并检查连词前后是否有谓语动词
        if token.pos_ == "CCONJ":
            left_has_verb = any(t.pos_ in {"VERB", "AUX"} for t in doc[temp_sent_start:token.i])
            right_has_verb = any(t.pos_ in {"VERB", "AUX"} for t in doc[token.i + 1:]) if token.i + 1 < len(doc) else False
            if left_has_verb and right_has_verb:
                # 当连词前后均有谓语时，根据连词分割句子
                if temp_sent_start < token.i:
                    sentences.append(doc[temp_sent_start:token.i].text.strip())
                temp_sent_start = token.i + 1

    # 添加最后一个段落作为句子，如果有的话
    if temp_sent_start < len(doc):
        sentences.append(doc[temp_sent_start:].text.strip())

    # 再次利用SpaCy的内置句子边界进行检测以确保每个部分都是完整的句子
    final_sentences = []
    for sentence in sentences:
        doc_temp = nlp(sentence)  # 对每个部分句子再次进行解析
        for sent in doc_temp.sents:
            final_sentences.append(sent.text.strip())  # 添加解析后的句子到最终列表

    return final_sentences



def process_text(text, aspect_keywords):
    doc = nlp(text)
    raw_sentences = split_sentences(doc)
    results = {}

    # 初始化每个类别的字典，用于聚合描述
    for category in aspect_keywords:
        results[category] = []

    # 分析每个句子
    for raw_sentence in raw_sentences:
        doc_sentence = nlp(raw_sentence)  # Re-parse each sentence
        for category, keywords in aspect_keywords.items():
            category_descriptions = {}  # 存储当前类别的名词和形容词描述
            for keyword in keywords:
                noun_adj_pairs = find_adjectives_modifying_nouns(doc_sentence, {keyword})
                for noun, adj in noun_adj_pairs:
                    noun_text = noun.text.lower()  # 将名词文本转为小写
                    if noun_text not in category_descriptions:
                        category_descriptions[noun_text] = []
                    structures = check_structure(doc_sentence, adj)
                    if structures:
                        category_descriptions[noun_text].extend(structures)
                    else:
                        category_descriptions[noun_text].append(adj.text)

            # 聚合当前类别下的所有描述，并添加到结果字典中
            for noun, descriptions in category_descriptions.items():
                full_description = f"{noun} is " + " and ".join(descriptions)
                results[category].append(full_description)

    # 为每个类别创建最终句子
    structured_results = []
    for category, descriptions in results.items():
        if descriptions:  # 只有当有描述时才加入结果
            category_sentence = f"{category.capitalize()}: " + ", ".join(descriptions) + "."
            structured_results.append(category_sentence)

    return structured_results

def read_and_process_file(file_path, aspect_keywords):
    # 打开文件并读取每一行
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 存储所有处理后的结果
    all_results = []

    # 处理每一行文本
    for line in lines:
        line = line.strip()  # 去掉每行的首尾空白字符
        if line:  # 确保不处理空行
            print("Processing line:", line)  # 打印正在处理的行，可选
            results = process_text(line, aspect_keywords)
            all_results.extend(results)  # 将结果添加到总结果列表中

    return all_results


# 调用函数，读取并处理文件
file_path = '../dataset/data.txt'  # 替换为您的文件路径
output = read_and_process_file(file_path, aspect_keywords)

# 打印所有处理结果
print("Identified complete structures from the file:")
for item in output:
    print(item)

