
from dotenv import load_dotenv
load_dotenv()

import threading
import logging
from grpc_server import serve as serve_grpc
from kafka_consumer import serve_kafka_consumer

logging.basicConfig(level=logging.INFO)

def main():
    grpc_thread = threading.Thread(target=serve_grpc, daemon=True)
    kafka_thread = threading.Thread(target=serve_kafka_consumer, daemon=True)
    grpc_thread.start()
    kafka_thread.start()
    grpc_thread.join()
    kafka_thread.join()

if __name__ == "__main__":
    main()
