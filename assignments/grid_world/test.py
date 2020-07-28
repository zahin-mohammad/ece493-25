import scipy.signal as signal
import numpy as np

def orig(rewards, discount):
    future_cumulative_reward = 0
    cumulative_rewards = np.empty_like(rewards, dtype=np.float64)
    for i in range(len(rewards) - 1, -1, -1):
        cumulative_rewards[i] = rewards[i] + discount * future_cumulative_reward
        future_cumulative_reward = cumulative_rewards[i]
    return cumulative_rewards

def alt(rewards, discount):
    """
    C[i] = R[i] + discount * C[i+1]
    signal.lfilter(b, a, x, axis=-1, zi=None)
    a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                          - a[1]*y[n-1] - ... - a[N]*y[n-N]
    """
    r = rewards[::-1]
    a = [1, -discount]
    b = [1]
    y = signal.lfilter(b, a, x=r)
    return y[::-1]

# test that the result is the same
np.random.seed(2017)

for i in range(100):
    rewards = np.random.random(10000)
    discount = 1.01
    expected = orig(rewards, discount)
    result = alt(rewards, discount)
    print(result)
    if not np.allclose(expected,result):
        print('FAIL: {}({}, {})'.format('alt', rewards, discount))
        break