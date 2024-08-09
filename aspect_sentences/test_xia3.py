import spacy
from transformers import AutoModel, AutoTokenizer

# 加载 transformer 模型和分词器
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer_model = AutoModel.from_pretrained(model_name)

# 加载 spaCy 的 transformer pipeline
nlp = spacy.blank("en")
nlp.add_pipe("transformer", config={"model": {"name": model_name}})

# 示例文本
text = "Barack Obama was born on August 4, 1961 in Honolulu, Hawaii."

# 使用 spaCy 和 transformer 模型解析文本
doc = nlp(text)

# 显示解析结果
for token in doc:
    print(f"Token: {token.text}, POS: {token.pos_}, DEP: {token.dep_}")

# 使用 spaCy 显示命名实体识别结果
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")

# 使用 displacy 可视化依存关系
from spacy import displacy
displacy.render(doc, style="dep")
