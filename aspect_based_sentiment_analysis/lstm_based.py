import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 读取数据
data = []
labels = []

with open('../aspect_sentences/dataset/data.txt', 'r') as f:
    for line in f:
        text, label = line.strip().split('\t')
        data.append(text)
        labels.append(1 if label == 'positive' else 0)  # 1表示正面情感，0表示负面情感

# 数据预处理
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=10)

# 创建LSTM模型
model = Sequential()
model.add(Embedding(len(word_index) + 1, 128, input_length=10))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(data, np.array(labels), epochs=10, batch_size=2)

# 预测
new_texts = ["The battery life is excellent."]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_data = pad_sequences(new_sequences, maxlen=10)
predictions = model.predict(new_data)
print(predictions)
