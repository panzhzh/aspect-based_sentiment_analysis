import webbrowser

import pandas as pd
import numpy as np
import spacy
from spacy import displacy
from spacy.util import minibatch, compounding
import explacy

#有命名实体的句子
#sentence = "Barack Obama was born on August 4, 1961 in Honolulu, Hawaii."

# 指定 CSV 文件的路径
file_path = '/Users/xiayan/Desktop/ana_sent/aspect_sentences/dataset/reviews_demo.csv'

# 读取 CSV 文件
data = pd.read_csv(file_path)
food_reviews_df = data[['review_text','rating']].dropna()


food_reviews_df.rating[food_reviews_df.rating<=4]=0
food_reviews_df.rating[food_reviews_df.rating>=5]=1

ax=food_reviews_df.rating.value_counts().plot(kind='bar')
fig = ax.get_figure()
fig.savefig("/Users/xiayan/Desktop/ana_sent/aspect_sentences/score_boolean.png")

print(food_reviews_df)

# 加载英语语言模型
spacy_tok = spacy.load("en_core_web_sm")
sample_review=food_reviews_df.Text[54]

#将 sample_review 变量中的文本数据使用 spacy_tok 函数进行处理
parsed_review = spacy_tok(sentence)
print(parsed_review)

#使用 explacy 库中的 print_parse_info 函数来可视化句子的依存关系和其他解析信息。
explacy.print_parse_info(spacy_tok, sentence)

tokenized_text = pd.DataFrame()

for i, token in enumerate(parsed_review):
    tokenized_text.loc[i, 'text'] = token.text
    tokenized_text.loc[i, 'lemma'] = token.lemma_,
    tokenized_text.loc[i, 'pos'] = token.pos_
    tokenized_text.loc[i, 'tag'] = token.tag_
    tokenized_text.loc[i, 'dep'] = token.dep_
    tokenized_text.loc[i, 'shape'] = token.shape_
    tokenized_text.loc[i, 'is_alpha'] = token.is_alpha
    tokenized_text.loc[i, 'is_stop'] = token.is_stop
    tokenized_text.loc[i, 'is_punctuation'] = token.is_punct
# 显示前20行
print(tokenized_text[:1])

#可视化解析结果中的命名实体（entities）例如人名、地名、组织名、日期等。
# 将命名实体可视化结果保存为 HTML 文件
html = displacy.render(parsed_review, style='ent')

# 保存 HTML 文件
output_path = "/Users/xiayan/Desktop/ana_sent/aspect_sentences/entity_visualization.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"实体可视化结果已保存到 {output_path}")
# 在默认浏览器中打开 HTML 文件
#webbrowser.open(f"file://{output_path}")

#将 parsed_review 对象中的句子分割并存储到一个列表中。
sentence_spans = list(parsed_review.sents)
print(sentence_spans)

noun_chunks_df = pd.DataFrame()

#将 parsed_review 对象中的名词短语（noun chunks）提取出来，并存储到一个 pandas DataFrame 中
for i, chunk in enumerate(parsed_review.noun_chunks):
    noun_chunks_df.loc[i, 'text'] = chunk.text
    noun_chunks_df.loc[i, 'root'] = chunk.root,
    noun_chunks_df.loc[i, 'root.text'] = chunk.root.text,
    noun_chunks_df.loc[i, 'root.dep_'] = chunk.root.dep_
    noun_chunks_df.loc[i, 'root.head.text'] = chunk.root.head.text

print(noun_chunks_df[:1])
