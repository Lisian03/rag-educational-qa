from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Generator:

    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, query, docs):

        context = "\n\n".join([
            f"[{i}] {doc.page_content}" for i, doc in enumerate(docs)
        ])

        prompt = f"""
Ti si studijski asistent za FER.

Odgovori jasno i edukativno koristeći SAMO kontekst.

Pitanje:
{query}

Kontekst:
{context}

Odgovor:
"""

        device = "cpu"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=300
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
