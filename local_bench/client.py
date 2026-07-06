import time
from dataclasses import dataclass

import requests


@dataclass
class ModelResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float


class LLMClient:
    def __init__(self, base_url: str = "http://localhost:11434/v1", api_key: str = "local"):
        if not base_url.rstrip("/").endswith("/v1"):
            import warnings
            warnings.warn(
                f"base_url '{base_url}' doesn't end with /v1 — "
                "this may not work with OpenAI-compatible servers"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def complete(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 120,
        extra_body: dict | None = None,
    ) -> ModelResponse:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            payload.update(extra_body)
        start = time.monotonic()
        try:
            resp = self._session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Connection refused — is your model server running? (expected at {self.base_url})"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Request timed out after {timeout}s — model may be loading or overloaded"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP error from model server: {e}")

        elapsed = time.monotonic() - start
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return ModelResponse(
            content=content,
            model=data.get("model", model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            elapsed_seconds=elapsed,
        )
