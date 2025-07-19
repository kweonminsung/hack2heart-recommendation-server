import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

def get_env(key: str, default=None):
    return os.environ.get(key, default)
