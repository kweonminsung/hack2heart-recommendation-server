
from dotenv import load_dotenv
import threading
import logging
import os
from grpc_server import serve as serve_grpc
from kafka_consumer import serve_kafka_consumer

# .env 파일 로드를 가장 먼저 실행
load_dotenv()

# 환경변수가 제대로 로드되었는지 확인용 로그
logging.basicConfig(level=logging.INFO)
logging.info(f"GRPC_PORT: {os.getenv('GRPC_PORT', 'NOT_SET')}")
logging.info(f"KAFKA_BOOTSTRAP_SERVERS: {os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'NOT_SET')}")
logging.info(f"KAFKA_TOPIC: {os.getenv('KAFKA_TOPIC', 'NOT_SET')}")

def main():
    grpc_thread = threading.Thread(target=serve_grpc, daemon=True)
    kafka_thread = threading.Thread(target=serve_kafka_consumer, daemon=True)
    
    grpc_thread.start()
    kafka_thread.start()
    
    try:
        # 메인 스레드를 살아있게 유지하면서 KeyboardInterrupt를 받을 수 있도록 함
        while grpc_thread.is_alive() or kafka_thread.is_alive():
            grpc_thread.join(timeout=1.0)  # 1초마다 체크
            kafka_thread.join(timeout=1.0)
    except KeyboardInterrupt:
        logging.info("Received interrupt signal, shutting down...")

if __name__ == "__main__":
    main()
