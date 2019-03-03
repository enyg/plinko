import numpy as np

# Environment
xfinal = np.random.randint(low=7, high=54, size=(3,))   # [Pos_R, Pos_G, Pos_B]
tfinal = np.random.rand(3)  # [T_R, T_G, T_B]
basket_velocity = 50    # 50 cm/sec
# Red (5 points), Green (4 points), and Blue (3 points)
scores = np.array([5, 4, 3])
# lastBasketPosition = np.random.randint(low=7, high=55)  # 20: "g20\n"
print("xfinal: {}".format(xfinal))
print("tfinal: {}".format(tfinal))

# Basic data structure
# [[0 0 0]
#  [1 0 0]
#  [0 1 0]
#  [0 0 1]
#  [0 1 1]
#  [1 0 1]
#  [1 1 0]
#  [1 1 1]] : 1 is select and 0 is unselect, 8 combinations in total
combination = np.zeros((8, 3), dtype=np.int)
for i in range(3):
    combination[i+1, i] = 1
    combination[i+4, i] = 1
combination[4:, :] = 1 - combination[4:, :]

scores_sorted = np.empty_like(scores)


# start processing
access = np.ones(8)

dt_move = np.zeros((3, 3))
dt_fall = np.zeros((3, 3))

idx_sorted = np.argsort(tfinal)
for i in range(3):
    scores_sorted[i] = scores[idx_sorted[i]]
    for j in range(3):
        if i == j:
            break
        dt_move[i, j] = np.abs(xfinal[idx_sorted[i]] - xfinal[idx_sorted[j]])
        dt_move[j, i] = dt_move[i, j]
        dt_fall[i, j] = np.abs(tfinal[idx_sorted[i]] - tfinal[idx_sorted[j]])
        dt_fall[j, i] = dt_fall[i, j]

scoreboard = combination * scores_sorted
scoreboard = scoreboard.sum(axis=1)
dt_move /= basket_velocity

# compare time interval
enough_time = (dt_move <= dt_fall)

# block impossible combinations
if not enough_time[0, 1]:
    access[6] = 0
    access[7] = 0
if not enough_time[1, 2]:
    access[4] = 0
    access[7] = 0
if not enough_time[0, 2]:
    access[5] = 0

# choose best strategy from possible combinations
combination_verified = combination[access > 0]
scoreboard_verified = scoreboard[access > 0]
idx_max_score = scoreboard_verified.argmax()
decision = combination_verified[idx_max_score]

# show the strategy
_map = {0: "Unselect", 1: "Select"}
print("1st ball: {}\n2nd ball: {}\n3rd ball: {}".format(
    _map[decision[0]], _map[decision[1]], _map[decision[2]]))
