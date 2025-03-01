from datetime import datetime
import pytz

def get_timestamp() -> str:
    """Get current timestamp in local timezone"""
    return datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M:%S %Z') 