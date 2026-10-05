"""Request/response schemas for the API."""

from pydantic import BaseModel, Field, field_validator

# Keep in sync with ReliableAgent.MAX_MESSAGE_CHARS.
MAX_MESSAGE_CHARS = 5000
# Longest allowed run of consecutive ' ' characters (tabs/newlines don't count).
MAX_SPACE_RUN = 200


class ChatRequest(BaseModel):
    """A single customer message. Identity comes from the API key, not the body."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_CHARS,
        description="The customer's raw message.",
    )

    @field_validator("message")
    @classmethod
    def check_message_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message must not be blank.")
        if " " * (MAX_SPACE_RUN + 1) in value:
            raise ValueError(f"Message must not contain more than {MAX_SPACE_RUN} spaces in a row.")
        return value


class ChatResponse(BaseModel):
    """The agent's reply for one turn."""

    user_id: str
    response: str


class AdminLoginRequest(BaseModel):
    """Admin credentials for password login."""

    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=200)


class TokenResponse(BaseModel):
    """The JWT returned after a successful admin login."""

    access_token: str
    token_type: str = "bearer"
