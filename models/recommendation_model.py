import numpy as np
from scipy.sparse import coo_matrix
from lightfm import LightFM
from lightfm.data import Dataset
import pickle
import os
import logging

class UserRecommendationModel:
    def __init__(self):
        self.dataset = None
        self.model = None
        self.interaction_matrix = None
        self.user_features = None
        self.is_trained = False

    def prepare_data(self, users, user_metadata, interactions):
        """데이터셋을 준비하고 행렬을 생성합니다."""
        print("=== 데이터셋 준비 중 ===")
        
        self.dataset = Dataset()
        self.dataset.fit(users, users)  # user → other user 추천

        all_features = set()
        for feats in user_metadata.values():
            all_features.update(feats)
        self.dataset.fit_partial(user_features=all_features)

        self.interaction_matrix, _ = self.dataset.build_interactions(interactions)
        user_feature_tuples = [(user, feats) for user, feats in user_metadata.items()]
        self.user_features = self.dataset.build_user_features(user_feature_tuples)

        print(f"데이터셋 준비 완료 - 사용자: {len(users)}, 피처 수: {len(all_features)}")

    def train_model(self, epochs=10, loss='bpr', num_threads=1):
        """모델을 학습시킵니다."""
        if self.dataset is None:
            raise ValueError("데이터셋이 준비되지 않았습니다. prepare_data()를 먼저 호출하세요.")
        
        print(f"=== 모델 학습 시작 - epochs: {epochs}, loss: {loss} ===")

        self.model = LightFM(loss=loss)
        self.model.fit(
            self.interaction_matrix,
            user_features=self.user_features,
            item_features=self.user_features,
            epochs=epochs,
            num_threads=num_threads
        )
        self.is_trained = True
        print("모델 학습 완료")

    def get_recommendations(self, user_id, top_n=10):
        """특정 사용자에게 추천할 사용자들을 반환합니다."""
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다. train_model()을 먼저 호출하세요.")

        print(f"=== 사용자 {user_id} 추천 계산 중 ===")

        try:
            user_idx = self.dataset.mapping()[0][user_id]
        except KeyError:
            raise ValueError(f"사용자 '{user_id}'를 찾을 수 없습니다.")

        user_ids = list(self.dataset.mapping()[0].keys())
        scores = self.model.predict(
            user_idx,
            np.arange(len(user_ids)),
            user_features=self.user_features,
            item_features=self.user_features
        )

        # 이미 알고 있는 사용자 제외
        known_positives = self.interaction_matrix.tocsr()[user_idx].indices
        # 추천 점수 정렬
        top_items = np.argsort(-scores)

        recommendations = []

        for i in top_items:
            if i == user_idx or i in known_positives:
                continue

            other_user = user_ids[i]
            score = float(scores[i])
            recommendations.append({
                'user_id': other_user,
                'score': score
            })

            if len(recommendations) >= top_n:
                break

        return recommendations

    def save_model(self, filepath):
        """모델을 파일로 저장합니다."""
        if not self.is_trained:
            raise ValueError("학습된 모델이 없습니다.")

        model_data = {
            'model': self.model,
            'dataset': self.dataset,
            'interaction_matrix': self.interaction_matrix,
            'user_features': self.user_features
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"모델이 {filepath}에 저장되었습니다.")

    def load_model(self, filepath):
        """파일에서 모델을 로드합니다."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {filepath}")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.dataset = model_data['dataset']
        self.interaction_matrix = model_data['interaction_matrix']
        self.user_features = model_data['user_features']
        self.is_trained = True

        print(f"모델이 {filepath}에서 로드되었습니다.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    from fetch_data import create_data
    
    load_dotenv()

    print("=== 스크립트 시작 ===")
    
    model = UserRecommendationModel()
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
    
    # 모델이 있으면 로드, 없으면 생성
    if os.path.exists(model_path):
        print("기존 모델을 로드합니다...")
        model.load_model(model_path)
        users = list(model.dataset.mapping()[0].keys())
    else:
        print("새 모델을 생성합니다...")
        users, user_metadata, interactions = create_data()
        model.prepare_data(users, user_metadata, interactions)
        model.train_model(epochs=10)
        model.save_model(model_path)

    user_id = users[0]

    recommendations = model.get_recommendations(user_id, top_n=10)

    print(f"\n🔎 사용자 {user_id}({user_metadata.get(user_id, [])})에게 추천:")
    for rec in recommendations:
        print(f"  👉 추천 대상: {rec_user_id} (예측 점수: {rec['score']:.8f}, 유저 정보: {user_metadata.get(rec['user_id'], [])}")

    print("=== 스크립트 종료 ===")
