import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize, sent_tokenize
from aspect_keywords import aspect_keywords


# 读取数据
def read_data(file_path):
    with open(file_path, 'r') as f:
        texts, labels = [], []
        for line in f:
            parts = line.strip().split('\t')
            texts.append(parts[0])
            labels.append(parts[1])
        return texts, labels


# 构建SVM模型
def build_svm_model():
    vectorizer = CountVectorizer()
    svm = SVC(kernel='linear', probability=True)
    model = make_pipeline(vectorizer, svm)
    return model


# 提取方面描述
def extract_aspect_sentences(text, aspect_keywords, model):
    sentences = sent_tokenize(text)
    aspect_sentences = {aspect: {'text': '', 'sentiment': 'neutral'} for aspect in aspect_keywords}

    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        prediction = model.predict([sentence])[0]
        sentiment = 'positive' if prediction == 1 else 'negative' if prediction == 0 else 'neutral'

        for aspect, keywords in aspect_keywords.items():
            if any(keyword in words for keyword in keywords):
                aspect_sentences[aspect]['text'] = sentence
                aspect_sentences[aspect]['sentiment'] = sentiment
                break
    return aspect_sentences


# 主函数
def main():
    data_path = 'data.txt'
    texts, labels = read_data(data_path)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    model = build_svm_model()
    model.fit(texts, y)

    test_text = "The food was great but the service was terrible. The ambiance was nice though."
    aspect_sentences = extract_aspect_sentences(test_text, aspect_keywords, model)
    print(aspect_sentences)


if __name__ == "__main__":
    main()
