from keras.datasets import boston_housing

(train, train_target), (test, test_target) = boston_housing.load_data()
print(train.shape)
print(test.shape)

"""
Scaling
(Standardization) 
"""
mean = train.mean(axis=0)
std = train.std(axis=0)
train -= mean
train /= std
test -= mean
test /= std

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

"""
데이터가 100개의 샘플에 의해 검증되므로, 어떻게 검증 데이터가 선택되었는지에 따라서 결과 값이 달라지게 됨
따라서, 교차 검증으로 해결함
일반적으로 K=4,5
"""

import numpy as np

k=4
num_val_samples = len(train) // k # 나눗셈 몫
num_epochs = 100
all_scores = []

for i in range(k):
    print('처리중인 폴드: ', i)
    val_data = train[i*num_val_samples: (i+1)*num_val_samples]
    val_target = train_target[i*num_val_samples: (i+1)*num_val_samples]

    partial_train_data = np.concatenate(
        [train[:i*num_val_samples], train[(i+1) * num_val_samples:]]
    , axis=0)

    partial_train_target = np.concatenate(
        [train_target[:i*num_val_samples], train_target[(i+1) * num_val_samples:]]
    , axis=0)

    model = build_model()
    model.fit(partial_train_data, partial_train_target, epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_target, verbose=0)
    all_scores.append(val_mae)

print(all_scores)


