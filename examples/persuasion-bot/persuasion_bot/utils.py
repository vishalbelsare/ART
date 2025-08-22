import random
import string


def generate_conversation_id() -> str:
    """Generates a unique 8 character conversation id."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=8))
