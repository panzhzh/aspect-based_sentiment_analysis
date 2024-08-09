import stanza

# 加载英语模型
nlp = stanza.Pipeline('en')

# 句子
sentence = "We had a fantastic meal at Armando al Pantheon! The food was delicious, especially the slow-roasted duck. The service was friendly, English-speaking, and efficient. The ambiance was warm, inviting, and comfortable, making for an enjoyable evening."

# 解析句子
doc = nlp(sentence)

# 打印句法依存关系和词性
for sent in doc.sentences:
    for word in sent.words:
        head = sent.words[word.head - 1].text if word.head > 0 else 'root'
        print(f"word: {word.text:10} head: {head:10} relation: {word.deprel:15} POS: {word.upos}")
