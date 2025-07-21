import grpc
from concurrent import futures
import time
import os
import proto.test_pb2 as test_pb2, proto.test_pb2_grpc as test_pb2_grpc
import proto.model_pb2 as model_pb2, proto.model_pb2_grpc as model_pb2_grpc
import logging
from models.recommendation_model import UserRecommendationModel
from models.fetch_data import create_data

GRPC_PORT = int(os.environ.get("GRPC_PORT", 50051))

# 전역 추천 모델 인스턴스
recommendation_model = None

def initialize_model():
    """추천 모델을 초기화합니다."""
    global recommendation_model
    try:
        recommendation_model = UserRecommendationModel()
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "saved_models", "recommendation_model.pkl")
        
        # 모델이 있으면 로드, 없으면 생성
        if os.path.exists(model_path):
            logging.info("기존 모델을 로드합니다...")
            recommendation_model.load_model(model_path)
        else:
            logging.info("새 모델을 생성합니다...")
            users, user_metadata, interactions = create_data()
            recommendation_model.prepare_data(users, user_metadata, interactions)
            recommendation_model.train_model(epochs=10)
            recommendation_model.save_model(model_path)
            
        logging.info("추천 모델 초기화 완료")
    except Exception as e:
        logging.error(f"추천 모델 초기화 실패: {e}")
        recommendation_model = None

class TestService(test_pb2_grpc.TestServiceServicer):
    def sendTestMessage(self, request, context):
        logging.info(f"Received message: {request.content}")
        return test_pb2.TestResponse(content=request.content)

class ModelService(model_pb2_grpc.ModelServiceServicer):
    def getUserRecommendations(self, request, context):
        logging.info(f"Received UserRequest for user_id: {request.user_id}")
        
        if recommendation_model is None:
            logging.error("추천 모델이 초기화되지 않았습니다")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("추천 모델이 초기화되지 않았습니다")
            return model_pb2.UserRecommendationsResponse()
        
        try:
            # 실제 추천 모델에서 추천 결과 가져오기
            model_recommendations = recommendation_model.get_recommendations(request.user_id, top_n=5)
            
            # protobuf 형식으로 변환
            recommendations = []
            for rec in model_recommendations:
                recommendation = model_pb2.Recommendation(
                    user_id=rec['user_id'],
                )
                recommendations.append(recommendation)
            
            return model_pb2.UserRecommendationsResponse(recommendations=recommendations)
            
        except ValueError as e:
            logging.error(f"사용자 추천 생성 오류: {e}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return model_pb2.UserRecommendationsResponse()
        except Exception as e:
            logging.error(f"예상치 못한 오류: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("내부 서버 오류")
            return model_pb2.UserRecommendationsResponse()

    def reGenerateModel(self, request, context):
        logging.info("Re-generating model...")
        
        global recommendation_model
        
        try:
            # 새로운 모델 인스턴스 생성
            recommendation_model = UserRecommendationModel()
            
            # 새로운 데이터로 모델 재학습
            users, user_metadata, interactions = create_data()
            recommendation_model.prepare_data(users, user_metadata, interactions)
            recommendation_model.train_model(epochs=10)
            
            # 모델 저장
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "saved_models", "recommendation_model.pkl")
            recommendation_model.save_model(model_path)
            
            logging.info("모델 재생성 완료")
            return model_pb2.Empty()
            
        except Exception as e:
            logging.error(f"모델 재생성 실패: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"모델 재생성 실패: {str(e)}")
            recommendation_model = None
            return model_pb2.Empty()

def serve():
    # 추천 모델 초기화
    initialize_model()
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    test_pb2_grpc.add_TestServiceServicer_to_server(TestService(), server)
    model_pb2_grpc.add_ModelServiceServicer_to_server(ModelService(), server)
    server.add_insecure_port(f'[::]:{GRPC_PORT}')
    server.start()
    logging.info(f"gRPC Server started on port {GRPC_PORT}")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)
