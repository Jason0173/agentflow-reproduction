import os, requests
from .base import EngineLM

class ModalEngine(EngineLM):
    def __init__(self, model_id: str = "", temperature: float = 0.0, **kwargs):
        self.url = os.environ["MODAL_PLANNER_URL"]
        self.temperature = temperature
        self.model_string = model_id

    def generate(self, prompt, system_prompt=None, **kwargs):
        if isinstance(prompt, str):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
            if all(isinstance(m, dict) for m in prompt):
                messages = prompt
            else:
                messages = [{"role": "user", "content": str(prompt[0])}]
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        # Ensure at least one user message exists
        has_user = any(m.get("role") == "user" for m in messages)
        if not has_user:
            content = messages[-1].get("content", str(prompt)) if messages else str(prompt)
            messages = [{"role": "user", "content": content}]

        try:
            resp = requests.post(
                self.url,
                json={"messages": messages, "temperature": self.temperature},
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling Modal endpoint: {str(e)}"

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
