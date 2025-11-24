from pydantic import BaseModel

class Message(BaseModel):
	text: str

class Feedback(BaseModel):
	text: str
	true_label: int  # 0 = ham, 1 = spam

