
from kafka import KafkaConsumer
import json
import os
import logging

KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "topic")
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")
KAFKA_GROUP_ID = os.environ.get("KAFKA_GROUP_ID", "consumer-group")

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
    except KeyboardInterrupt:
        logging.info("Consumer stopped.")
    finally:
        consumer.close()
