import grpc
from concurrent import futures
import time
import os
import proto.test_pb2 as test_pb2, proto.test_pb2_grpc as test_pb2_grpc
import proto.model_pb2 as model_pb2, proto.model_pb2_grpc as model_pb2_grpc
import logging
from models.recommendation_model import global_model as recommendation_model, MODEL_PATH
from models.fetch_data import create_data

GRPC_PORT = int(os.environ.get("GRPC_PORT", 50051))

def initialize_model():
    global recommendation_model
    try:
        if os.path.exists(MODEL_PATH):
            logging.info("Loading existing model")
            recommendation_model.load_model()
        else:
            logging.info("Creating new model")
            users, user_metadata, interactions = create_data()
            recommendation_model.prepare_data(users, user_metadata, interactions)
            recommendation_model.train_model(epochs=10)
            recommendation_model.save_model()
        logging.info("Completed model initialization")
    except Exception as e:
        logging.error(f"Error initializing recommendation model: {e}")

class TestService(test_pb2_grpc.TestServiceServicer):
    def sendTestMessage(self, request, context):
        logging.info(f"Received message: {request.content}")
        return test_pb2.TestResponse(content=request.content)

class ModelService(model_pb2_grpc.ModelServiceServicer):
    def getUserRecommendations(self, request, context):
        logging.info(f"Received UserRequest for user_id: {request.user_id}, type: {type(request.user_id)}")
        try:
            # 실제 추천 모델에서 추천 결과 가져오기
            model_recommendations = recommendation_model.get_recommendations(request.user_id, top_n=5)
            # protobuf 형식으로 변환
            user_ids = [rec['user_id'] for rec in model_recommendations]
            return model_pb2.UserRecommendationsResponse(user_ids=user_ids)
        except ValueError as e:
            logging.error(f"Error generating recommendations for user {request.user_id}: {e}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return model_pb2.UserRecommendationsResponse(user_ids=[])
        except Exception as e:
            logging.error(f"Internal server error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return model_pb2.UserRecommendationsResponse(user_ids=[])

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
