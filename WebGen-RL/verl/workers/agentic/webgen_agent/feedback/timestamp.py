from datetime import datetime
from zoneinfo import ZoneInfo

def current_timestamp() -> str:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    return now.strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3]


if __name__ == "__main__":
    print(current_timestamp())  # Example usage
