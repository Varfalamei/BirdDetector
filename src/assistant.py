import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from default_system import DEFAULT_SYSTEM_PROMPT


MODEL_NAME = "model_llm/model"
MODEL_TOKEN = "model_llm/tokenizer"
MODEL_GENERATION_CONFIG = "model_llm/generation_config"
DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>"
DEFAULT_RESPONSE_TEMPLATE = "<s>bot\n"


config = PeftConfig.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, torch_dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(model, MODEL_NAME, torch_dtype=torch.float16)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKEN, use_fast=False)
generation_config = GenerationConfig.from_pretrained(MODEL_GENERATION_CONFIG)


class Conversation:
    def init(
        self,
        message_template=DEFAULT_MESSAGE_TEMPLATE,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_template=DEFAULT_RESPONSE_TEMPLATE,
    ):
        self.message_template = message_template
        self.response_template = response_template
        self.messages = [{"role": "system", "content": system_prompt}]

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def add_bot_message(self, message):
        self.messages.append({"role": "bot", "content": message})

    def get_prompt(self, tokenizer):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += DEFAULT_RESPONSE_TEMPLATE
        return final_text.strip()


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(**data, generation_config=generation_config)[0]
    output_ids = output_ids[len(data["input_ids"][0]) :]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()
