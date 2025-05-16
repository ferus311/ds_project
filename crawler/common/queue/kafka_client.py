from confluent_kafka import Producer, Consumer, KafkaError
import json
import socket
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class KafkaProducer:
    def __init__(self, bootstrap_servers=None):
        if bootstrap_servers is None:
            # Lấy từ biến môi trường hoặc mặc định
            bootstrap_servers = os.environ.get(
                "KAFKA_BOOTSTRAP_SERVERS", "kafka1:19092"
            )

        self.producer = Producer(
            {"bootstrap.servers": bootstrap_servers, "client.id": socket.gethostname()}
        )

    def send(self, topic, value, key=None):
        """Gửi message tới Kafka topic"""
        try:
            # Nếu value là dict, chuyển đổi thành JSON string
            if isinstance(value, dict):
                value = json.dumps(value)

            # Gửi message đến Kafka
            self.producer.produce(
                topic=topic,
                key=key,
                value=value.encode("utf-8") if isinstance(value, str) else value,
            )
            self.producer.flush()
            logger.debug(f"Sent message to topic {topic}")
            return True
        except Exception as e:
            logger.error(f"Error sending message to Kafka: {e}")
            return False


class KafkaConsumer:
    def __init__(self, topics, group_id, bootstrap_servers=None):
        if bootstrap_servers is None:
            bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka1:19092")

        self.consumer = Consumer({
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True
        })
        self.consumer.subscribe(topics)
        logger.info(f"Kafka Consumer subscribed to topics: {topics}")

    def consume(self, timeout=1.0):
        """Tiêu thụ message từ Kafka"""
        try:
            msg = self.consumer.poll(timeout)

            if msg is None:
                return None

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.debug('Reached end of partition')
                else:
                    logger.error(f'Error while consuming: {msg.error()}')
                return None

            # Decode giá trị message
            try:
                value = msg.value().decode('utf-8')
                return json.loads(value)
            except json.JSONDecodeError:
                return value
            except Exception as e:
                logger.error(f"Error decoding message: {e}")
                return msg.value()

        except Exception as e:
            logger.error(f"Error consuming message from Kafka: {e}")
            return None

    def close(self):
        """Đóng kết nối consumer"""
        self.consumer.close()
