import grpc
from concurrent import futures
import time
import proto.test_pb2 as test_pb2, proto.test_pb2_grpc as test_pb2_grpc
import logging

class TestService(test_pb2_grpc.TestServiceServicer):
    def sendTestMessage(self, request, context):
        logging.info(f"Received message: {request.content}")
        return test_pb2.TestResponse(content=request.content)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    test_pb2_grpc.add_TestServiceServicer_to_server(TestService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logging.info("gRPC Server started on port 50051")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
