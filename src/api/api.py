import torch
from fastapi import FastAPI
from src.api.model import PostGetRequestFromModel

from src.assistant import Conversation, generate, tokenizer, model, generation_config

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "I'm health"}


@app.post(
    "/predict",
    description="""
    сделать предикт
    """,
    response_model=PostGetRequestFromModel,
)
def get_predictions_by_model(body: PostGetRequestFromModel) -> PostGetRequestFromModel:
    message_input = body.message

    conversation = Conversation()
    conversation.add_user_message(message_input)
    prompt = conversation.get_prompt(tokenizer)

    message_output = generate(model, tokenizer, prompt, generation_config)

    return PostGetRequestFromModel(message=message_output)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8007, reload=True)
