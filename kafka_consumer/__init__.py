
from kafka import KafkaConsumer
import json
import os
import logging
from models.recommendation_model import global_model
from models.fetch_data import create_data

KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "topic")
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")
KAFKA_GROUP_ID = os.environ.get("KAFKA_GROUP_ID", "consumer-group")

def regenerate_model():
    global global_model
    users, user_metadata, interactions = create_data()
    global_model.prepare_data(users, user_metadata, interactions)
    global_model.train_model(epochs=10)
    global_model.save_model()
    logging.info("모델이 재생성되었습니다.")


def insert_user_reaction(user_id: int, target_user_id: int, reaction_weight: float = 1.0):
    """사용자 반응을 기존 모델에 삽입합니다."""
    global global_model
    try:
        # 모델이 학습되어 있지 않으면 로드 시도
        if not global_model.is_trained:
            global_model.load_model()
            logging.info(f"모델 로드 완료")
        # 사용자 반응 업데이트
        global_model.update_user_reaction(user_id, target_user_id, reaction_weight)
        # 업데이트된 모델 저장
        global_model.save_model()
        logging.info(f"사용자 반응 삽입 완료: {user_id} → {target_user_id} (가중치: {reaction_weight})")
    except Exception as e:
        logging.error(f"사용자 반응 삽입 중 오류: {e}")


def serve_kafka_consumer():
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id=KAFKA_GROUP_ID,
        value_deserializer=lambda m: json.loads(m.decode("utf-8"))
    )
    logging.info(f"Kafka consumer started. Listening to topic: {KAFKA_TOPIC}")
    try:
        for message in consumer:
            logging.info(f"Received message: {message.value}")
            # 메시지 수신 시 모델 재생성
            # regenerate_model()
    except KeyboardInterrupt:
        logging.info("Consumer stopped.")
    finally:
        consumer.close()
