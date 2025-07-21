
from dotenv import load_dotenv
import threading
import logging
from grpc_server import serve as serve_grpc
from kafka_consumer import serve_kafka_consumer

load_dotenv()

logging.basicConfig(level=logging.INFO)

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
