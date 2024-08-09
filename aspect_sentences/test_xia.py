import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 首先确保已经下载了 VADER 的词典数据
nltk.download('vader_lexicon')

# 初始化情感分析器
sia = SentimentIntensityAnalyzer()

# 获取 VADER 词典
vader_dict = sia.lexicon

# 打印词典中的前几项，查看单词和其情感分数
for word, score in list(vader_dict.items())[:10]:
    print(f"Word: {word}, Score: {score}")
