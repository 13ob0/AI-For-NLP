"""

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

content = pd.read_csv('../DataSource/train.csv')
content = content.dropna()

age_with_fares = content[
    (content['Age'] > 22) & (content['Fare'] < 400) & (content['Fare'] > 130)
]

sub_fare = age_with_fares['Fare']
sub_age = age_with_fares['Age']


def func(age, k, b): return k * age + b


def loss(y, yhat):
    """

    :param y: the real fares
    :param yhat: the estimated fares
    :return: how good is the estimated fares
    """
    # return np.mean(np.abs(y - yhat))
    # return np.mean(np.square(y - yhat))
    return np.mean(np.sqrt(y - yhat))


min_error_rate = float('inf')

loop_times = 10000

losses = []

change_directions = [
    # (k, b)
    (+1, -1),  # k increase, b decrease
    (+1, +1),
    (-1, +1),
    (-1, -1)  # k decrease, b decrease
]

k_hat = random.random() * 20 - 10
b_hat = random.random() * 20 - 10

best_k, best_b = k_hat, b_hat

best_direction = None


def step(): return random.random() * 1


direction = random.choice(change_directions)

def derivate_k(y, yhat, x):
   abs_values =  [1 if (y_i - yhat_i) > 0 else -1 for y_i, yhat_i in zip(y, yhat)]
   return np.mean([a * -x_i for a, x_i in zip(abs_values, x)])

def derivate_b(y, yhat):
    abs_values = [1 if (y_i - yhat_i) > 0 else -1 for y_i, yhat_i in zip(y, yhat)]
    return np.mean([a * -1 for a in abs_values])

while loop_times > 0:

    k_delta_direction, b_delta_direction = direction

    k_delta = k_delta_direction * step()
    b_delta = b_delta_direction * step()

    new_k = best_k + k_delta
    new_b = best_b + b_delta

    estimated_fares = func(sub_age, new_k, new_b)
    error_rate = loss(y=sub_fare, yhat=estimated_fares)

    if error_rate < min_error_rate:
        min_error_rate = error_rate
        best_k, best_b = new_k, new_b

        direction = (k_delta_direction, b_delta_direction)
        print(min_error_rate)
        print('loop == {}'.format(10000 - loop_times))
        losses.append(min_error_rate)
        print('f(age) = {} * age + {}, with error rate: {}'.format(best_k,
                                                                   best_b, error_rate))
    else:
        direction = random.choice(list(set(change_directions) - {(k_delta_direction, b_delta_direction)}))

    loop_times -= 1

plt.scatter(sub_age, sub_fare)
plt.plot(sub_age, func(sub_age, best_k, best_b), c='r')
plt.show()

# plt.plot(range(len(losses)), losses)
# plt.show()
