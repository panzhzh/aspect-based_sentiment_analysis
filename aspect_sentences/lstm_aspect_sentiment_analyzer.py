import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
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


# 数据预处理
def preprocess_data(texts, max_length, tokenizer):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences


# 构建LSTM模型
def build_lstm_model(vocab_size, max_length):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # 3 classes: positive, negative, neutral
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 提取形容词
def extract_adjectives(tokens):
    tagged = pos_tag(tokens)
    adjectives = [word for word, pos in tagged if pos.startswith('JJ')]
    return adjectives


# 提取方面描述并进行情感分析
def extract_aspect_sentences(text, aspect_keywords, model, tokenizer, max_length):
    sentences = sent_tokenize(text)
    aspect_sentences = {aspect: {'text': '', 'sentiment': 'neutral'} for aspect in aspect_keywords}

    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        adjectives = extract_adjectives(words)

        for aspect, keywords in aspect_keywords.items():
            if any(keyword in words for keyword in keywords):
                for adj in adjectives:
                    adj_index = words.index(adj)
                    relevant_part = " ".join(words[:adj_index + 1])
                    sequence = tokenizer.texts_to_sequences([relevant_part])
                    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
                    prediction = model.predict(padded_sequence)[0]
                    sentiment = np.argmax(prediction)
                    sentiment_label = 'positive' if sentiment == 1 else 'negative' if sentiment == 0 else 'neutral'

                    aspect_sentences[aspect]['text'] = relevant_part
                    aspect_sentences[aspect]['sentiment'] = sentiment_label
                    break

    return aspect_sentences


# 主函数
def main():
    data_path = 'data.txt'
    texts, labels = read_data(data_path)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = 100

    X = preprocess_data(texts, max_length, tokenizer)
    y = tf.keras.utils.to_categorical(labels, num_classes=3)

    model = build_lstm_model(vocab_size, max_length)
    model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

    test_text = "The food was great but the service was terrible. The ambiance was nice though."
    aspect_sentences = extract_aspect_sentences(test_text, aspect_keywords, model, tokenizer, max_length)
    print(aspect_sentences)


if __name__ == "__main__":
    main()
