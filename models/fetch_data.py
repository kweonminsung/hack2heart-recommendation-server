import random
import mysql.connector
import os
import logging
from typing import Tuple, List, Dict, Optional


def get_db_connection():
    """MySQL 데이터베이스 연결 생성"""
    try:
        connection = mysql.connector.connect(
            host=os.environ.get('DB_HOST', 'localhost'),
            port=int(os.environ.get('DB_PORT', 3306)),
            user=os.environ.get('DB_USER', ''),
            password=os.environ.get('DB_PASSWORD', ''),
            database=os.environ.get('DB_NAME', ''),
            charset='utf8mb4'
        )
        return connection
    except mysql.connector.Error as err:
        logging.error(f"MySQL 연결 실패: {err}")
        raise


def create_data() -> Optional[Tuple[List[str], Dict[str, List[str]], List[Tuple[str, str, int]]]]:
    """
    MySQL에서 유저 feature 및 interaction 데이터를 로딩

    Returns:
        users: 유저 ID 리스트
        user_metadata: {user_id: [feature list]}
        interactions: (from_user_id, to_user_id, rating) 튜플 리스트
    """
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        # 유저 정보 조회 쿼리
        cursor.execute("""
            WITH OrderedTmi AS (
                SELECT
                    user_id,
                    tmi_id,
                    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY UserTmi.id) AS rn
                FROM UserTmi
            )
            SELECT
                User.id,
                User.gender,
                User.looking_for_love,
                User.looking_for_friend,
                User.looking_for_coworker,
                Language.name AS most_preferred_language,
                Package.name AS most_preferred_package,
                Tmi1.name AS first_tmi,
                Tmi2.name AS second_tmi,
                Tmi3.name AS third_tmi
            FROM User
            JOIN Language ON User.most_preferred_language_id = Language.id
            JOIN Package ON User.most_preferred_package_id = Package.id
            LEFT JOIN OrderedTmi AS ot1 ON ot1.user_id = User.id AND ot1.rn = 1
            LEFT JOIN OrderedTmi AS ot2 ON ot2.user_id = User.id AND ot2.rn = 2
            LEFT JOIN OrderedTmi AS ot3 ON ot3.user_id = User.id AND ot3.rn = 3
            LEFT JOIN Tmi AS Tmi1 ON Tmi1.id = ot1.tmi_id
            LEFT JOIN Tmi AS Tmi2 ON Tmi2.id = ot2.tmi_id
            LEFT JOIN Tmi AS Tmi3 ON Tmi3.id = ot3.tmi_id;
        """)
        user_rows = cursor.fetchall()

        users: List[str] = []
        user_metadata: Dict[str, List[str]] = {}

        for row in user_rows:
            user_id = row['id']
            users.append(user_id)

            features = [
                f"gender={row['gender']}" if row['gender'] else '',
                'looking_for=love' if row['looking_for_love'] else '',
                'looking_for=friend' if row['looking_for_friend'] else '',
                'looking_for=coworker' if row['looking_for_coworker'] else '',
                f"language={row['most_preferred_language']}" if row['most_preferred_language'] else '',
                f"package={row['most_preferred_package']}" if row['most_preferred_package'] else '',
                f"tmi={row['first_tmi']}" if row['first_tmi'] else '',
                f"tmi={row['second_tmi']}" if row['second_tmi'] else '',
                f"tmi={row['third_tmi']}" if row['third_tmi'] else ''
            ]
            user_metadata[user_id] = [f for f in features if f]

        # 유저 간 상호작용 조회
        cursor.execute("""
            SELECT
                from_user_id, to_user_id,
                CASE type
                    WHEN 'SUPER_LIKE' THEN 3
                    WHEN 'LIKE' THEN 2
                    WHEN 'DISLIKE' THEN 0
                END AS rating
            FROM UserReaction
        """)

        reaction_rows = cursor.fetchall()
        interactions: List[Tuple[str, str, int]] = [
            (row['from_user_id'], row['to_user_id'], row['rating'])
            for row in reaction_rows if row['rating'] is not None
        ]

        logging.info(f"MySQL에서 데이터 로드 완료 - 사용자: {len(users)}, 상호작용: {len(interactions)}")
        cursor.close()
        connection.close()

        return users, user_metadata, interactions

    except mysql.connector.Error as err:
        logging.error(f"MySQL 쿼리 실행 오류: {err}")
        return [[], {}, []]
    except Exception as err:
        logging.error(f"예상치 못한 오류: {err}")
        return [[], {}, []]

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    
    result = create_data()
    if result:
        users, metadata, interactions = result
        print(f"Loaded {len(users)} users and {len(interactions)} interactions.")
