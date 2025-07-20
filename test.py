import numpy as np
from scipy.sparse import coo_matrix
from lightfm import LightFM
import pickle

# 사용자-아이템 상호작용 행렬
# 예: user_id, item_id, interaction(1=like, 0=skip)
interactions = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
    [2, 2, 1],
    [2, 0, 1],
    [3, 3, 1],
])

user_ids = interactions[:, 0]
item_ids = interactions[:, 1]
ratings = interactions[:, 2]

# sparse matrix 생성
interaction_matrix = coo_matrix((ratings, (user_ids, item_ids)))

# LightFM 모델 학습
model = LightFM(loss='warp')
model.fit(interaction_matrix, epochs=10, num_threads=2)

# 모델 저장
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# sparse matrix 저장
with open("matrix.pkl", "wb") as f:
    pickle.dump(interaction_matrix, f)
