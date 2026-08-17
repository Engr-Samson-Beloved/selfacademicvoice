from pydantic import BaseModel


class RewriteRequest(BaseModel):
    draft: str
    system_prompt_override: str | None = None


class RewriteResponse(BaseModel):
    rewritten: str


class UploadResponse(BaseModel):
    filename: str
    rewritten: str
