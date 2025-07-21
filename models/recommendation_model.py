import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from lightfm import LightFM
from lightfm.data import Dataset
import pickle
import os
from typing import List, Dict, Tuple, Optional, Any
import logging

# 전역 모델 인스턴스 및 경로 (다른 모듈에서 import해서 재사용)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

class UserRecommendationModel:
    """사용자 간 추천을 위한 LightFM 기반 모델 클래스"""
    
    def __init__(self):
        self.dataset: Optional[Dataset] = None
        self.model: Optional[LightFM] = None
        self.interaction_matrix: Optional[csr_matrix] = None
        self.user_features: Optional[csr_matrix] = None
        self.is_trained: bool = False

    def prepare_data(self, users: List[int], user_metadata: Dict[int, List[str]], 
                    interactions: List[Tuple[int, int, float]]) -> None:
        """데이터셋을 준비하고 행렬을 생성합니다."""
        logging.info("Preparing dataset with users and interactions")
        
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

    def train_model(self, epochs: int = 10, loss: str = 'warp-kos', num_threads: int = 1) -> None:
        """모델을 학습시킵니다."""
        if self.dataset is None:
            raise ValueError("Dataset is not prepared. Call prepare_data() first.")
        
        logging.info(f"Training model - epochs: {epochs}, loss: {loss}")

        self.model = LightFM(loss=loss)
        self.model.fit(
            self.interaction_matrix,
            user_features=self.user_features,
            item_features=self.user_features,
            epochs=epochs,
            num_threads=num_threads
        )
        self.is_trained = True
        logging.info("Model training completed")

    def get_recommendations(self, user_id: int, top_n: int = 10) -> List[Dict[int, Any]]:
        """특정 사용자에게 추천할 사용자들을 반환합니다."""
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train_model() first.")

        logging.info(f"Calculating recommendations for user: {user_id}")

        try:
            user_idx = self.dataset.mapping()[0][user_id]
        except KeyError:
            raise ValueError(f"User '{user_id}' not found in dataset.")

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

    def save_model(self) -> None:
        """모델을 파일로 저장합니다."""
        if not self.is_trained:
            logging.warning("No trained model to save.")

        model_data = {
            'model': self.model,
            'dataset': self.dataset,
            'interaction_matrix': self.interaction_matrix,
            'user_features': self.user_features
        }

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model_data, f)
        logging.info(f"Model saved to {MODEL_PATH}")

    def load_model(self) -> None:
        """파일에서 모델을 로드합니다."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.dataset = model_data['dataset']
        self.interaction_matrix = model_data['interaction_matrix']
        self.user_features = model_data['user_features']
        self.is_trained = True

        logging.info(f"Model loaded from {MODEL_PATH}")

    def _validate_user_ids(self, user_id: int, target_user_id: int) -> Tuple[int, int]:
        """사용자 ID 유효성 검사 및 인덱스 반환"""
        user_mapping = self.dataset.mapping()[0]
        missing_users = []
        
        if user_id not in user_mapping:
            missing_users.append(user_id)
        if target_user_id not in user_mapping:
            missing_users.append(target_user_id)
            
        if missing_users:
            raise ValueError(f"Users not found: {', '.join(missing_users)}")
        
        return user_mapping[user_id], user_mapping[target_user_id]

    def _update_interaction_matrix(self, rows: List[int], cols: List[int], data: List[float]) -> None:
        """상호작용 매트릭스 업데이트"""
        current_matrix = self.interaction_matrix.tocoo()
        
        # 기존 데이터와 새 데이터 결합
        all_rows = list(current_matrix.row) + rows
        all_cols = list(current_matrix.col) + cols
        all_data = list(current_matrix.data) + data
        
        # 새로운 interaction matrix 생성
        self.interaction_matrix = coo_matrix(
            (all_data, (all_rows, all_cols)), 
            shape=current_matrix.shape
        ).tocsr()

    def _retrain_model(self, epochs: int = 1) -> None:
        """모델 재학습 (partial_fit 시도 후 실패 시 전체 재학습)"""
        try:
            self.model.fit_partial(
                self.interaction_matrix,
                user_features=self.user_features,
                item_features=self.user_features,
                epochs=epochs,
                num_threads=1
            )
        except Exception as e:
            logging.warning(f"Partial fit failed: {e}, retraining model")
            self.model.fit(
                self.interaction_matrix,
                user_features=self.user_features,
                item_features=self.user_features,
                epochs=epochs,
                num_threads=1
            )

    def update_user_reaction(self, user_id: int, target_user_id: int, reaction_weight: float = 1.0) -> None:
        """새로운 사용자 반응을 모델에 반영합니다."""
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train_model() first.")
        
        logging.info(f"Updating user reaction: {user_id} → {target_user_id} (weight: {reaction_weight})")
        
        try:
            user_idx, target_user_idx = self._validate_user_ids(user_id, target_user_id)
            self._update_interaction_matrix([user_idx], [target_user_idx], [reaction_weight])
            self._retrain_model()
            
            logging.info(f"Reaction update completed: {user_id} → {target_user_id}")
                
        except Exception as e:
            logging.error(f"Error updating reaction: {e}")

    def batch_update_reactions(self, reactions_list: List[Tuple[int, int, float]]) -> None:
        """여러 사용자 반응을 배치로 업데이트합니다."""
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train_model() first.")
            
        if not reactions_list:
            logging.info("No reactions to update.")
            return
            
        logging.info(f"Batch updating {len(reactions_list)} reactions")
        
        try:
            user_mapping = self.dataset.mapping()[0]
            rows, cols, data = [], [], []
            valid_reactions = 0
            
            for user_id, target_user_id, weight in reactions_list:
                if user_id in user_mapping and target_user_id in user_mapping:
                    user_idx = user_mapping[user_id]
                    target_user_idx = user_mapping[target_user_id]
                    
                    rows.append(user_idx)
                    cols.append(target_user_idx)
                    data.append(weight)
                    valid_reactions += 1
                else:
                    logging.warning(f"Skipping reaction due to missing user: {user_id} → {target_user_id}")
            
            if valid_reactions == 0:
                logging.info("No valid reactions to update.")
                return
                
            self._update_interaction_matrix(rows, cols, data)
            self._retrain_model()
            
            logging.info(f"Batch update completed: {valid_reactions} reactions processed")
                
        except Exception as e:
            raise ValueError(f"Batch reaction update failed: {str(e)}")

global_model = UserRecommendationModel()

if __name__ == "__main__":
    from dotenv import load_dotenv
    from fetch_data import create_data
    
    load_dotenv()

    print("=== 스크립트 시작 ===")
    
    model = UserRecommendationModel()
    users, user_metadata, interactions = create_data()
    model.prepare_data(users, user_metadata, interactions)
    model.train_model(epochs=10)
    model.save_model()

    # 추천 테스트
    for idx in range(min(10, len(users))):
        user_id = users[idx]
        try:
            recommendations = model.get_recommendations(user_id, top_n=5)
            
            print(f"\n🔎 사용자 {user_id}({user_metadata.get(user_id, [])})에게 추천:")
            for rec in recommendations:
                rec_user_id = rec['user_id']
                print(f"  👉 {rec_user_id} (점수: {rec['score']:.6f}, 정보: {user_metadata.get(rec_user_id, [])})")
        except Exception as e:
            print(f"사용자 {user_id} 추천 중 오류: {e}")

    print("=== 스크립트 종료 ===")
