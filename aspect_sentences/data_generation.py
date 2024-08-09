# 示例数据（包含正面和负面标签）
data = [
    "Sisig (so far the best in KL) and halo-halo tastes so good. Definitely authentic Filipino food. We enjoyed dining at their resto. And the staff are very friendly too. We will certainly come back.",
    "Good Filipino food! Tortang talong was a winner. Service is good and staff are all friendly. Will come back again to try out other dishes on the menu.",
    "Looking for a calm place to chill yourself after a long hectic day, this place has it all. Great Filipino food, and as usual some liquor to calm you down.",
    "Good place for friends and family..refreshing drinks.. Good vibes. And service is very good...mostly if you want a chill place for meeting and relaxing this is the best choice..thanks a lot",
    "Food is good and so is the ambiance of the Resto. They have a billiard table and karaoke in-house as well. You'll love it here!",
    "The food was terrible and the service was very slow. I will not come back.",
    "Awful experience! The ambiance was too noisy and the staff were rude.",
    "The food is overpriced and not tasty. The service is mediocre at best.",
    "The place is dirty and the food quality is very low. Not recommended.",
    "The food was cold and the service was terrible. We had a bad time."
]

labels = ['positive', 'positive', 'positive', 'positive', 'positive', 'negative', 'negative', 'negative', 'negative', 'negative']

# 将数据保存到 data.txt 文件中
with open('data.txt', 'w') as f:
    for text, label in zip(data, labels):
        f.write(f"{text}\t{label}\n")
