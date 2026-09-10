"""LLM providers: OpenAI and Gemini, both returning Pydantic models."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Type, TypeVar

import google.generativeai as genai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from src import config

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)

LLM_MAX_ATTEMPTS = 3


class LLMError(Exception):
    pass


def _is_permanent_llm_error(exc: LLMError) -> bool:
    text = str(exc).lower()
    return "missing or empty" in text or "must be 'openai' or 'gemini'" in text


def _missing_key_message(key_name: str, provider: str) -> str:
    other = "gemini" if provider == "openai" else "openai"
    other_ready = (
        config.gemini_key_ready() if provider == "openai" else config.openai_key_ready()
    )
    message = (
        f"{key_name} is missing or empty. Add it to .env next to app.py, "
        "then restart Streamlit."
    )
    if other_ready:
        return (
            f"{message} A {other} key is loaded — choose {other} in the sidebar "
            "if you want to use that instead."
        )
    return (
        f"{message} You only need the key for the provider you select "
        f"(openai or gemini)."
    )


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise LLMError("The model did not return valid JSON.")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise LLMError("The model JSON was not an object.")
    return data


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        config.reload_env()
        self.provider = (provider or config.LLM_PROVIDER).strip().lower()
        if self.provider not in {"openai", "gemini"}:
            raise LLMError("Provider must be 'openai' or 'gemini'.")

        self._lock = threading.Lock()
        if self.provider == "openai":
            if not config.openai_key_ready():
                raise LLMError(_missing_key_message("OPENAI_API_KEY", "openai"))
            self.model = model or config.OPENAI_MODEL
            self._openai = OpenAI(api_key=config.OPENAI_API_KEY)
        else:
            if not config.gemini_key_ready():
                raise LLMError(_missing_key_message("GEMINI_API_KEY", "gemini"))
            self.model = model or config.GEMINI_MODEL
            genai.configure(api_key=config.GEMINI_API_KEY)
            self._gemini = genai.GenerativeModel(self.model)

    def generate_structured(
        self,
        schema: Type[T],
        system_prompt: str,
        user_prompt: str,
    ) -> T:
        schema_hint = json.dumps(schema.model_json_schema(), indent=2)
        full_user = (
            f"{user_prompt}\n\n"
            "Return ONLY valid JSON that matches this schema:\n"
            f"{schema_hint}"
        )
        raw = self._complete(system_prompt, full_user, json_mode=True)
        try:
            payload = parse_json_object(raw)
            return schema.model_validate(payload)
        except (LLMError, ValidationError, json.JSONDecodeError) as exc:
            raise LLMError(f"Could not parse structured output: {exc}") from exc

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Free-text completion for grounded chat. Does not require JSON."""
        return self._complete(system_prompt, user_prompt, json_mode=False)

    def _complete(self, system_prompt: str, user_prompt: str, json_mode: bool = True) -> str:
        last_error: LLMError | None = None
        for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
            try:
                if self.provider == "openai":
                    return self._complete_openai(system_prompt, user_prompt, json_mode=json_mode)
                return self._complete_gemini(system_prompt, user_prompt, json_mode=json_mode)
            except LLMError as exc:
                last_error = exc
                if _is_permanent_llm_error(exc) or attempt == LLM_MAX_ATTEMPTS:
                    raise
                logger.warning(
                    "LLM call failed (attempt %s/%s): %s",
                    attempt,
                    LLM_MAX_ATTEMPTS,
                    exc,
                )
                time.sleep(0.6 * attempt)
        raise last_error or LLMError("LLM request failed.")

    def _complete_openai(self, system_prompt: str, user_prompt: str, json_mode: bool) -> str:
        try:
            kwargs: dict = {
                "model": self.model,
                "temperature": 0.1 if json_mode else 0.2,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = self._openai.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            if not content.strip():
                raise LLMError("OpenAI returned an empty response.")
            return content
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc

    def _complete_gemini(self, system_prompt: str, user_prompt: str, json_mode: bool) -> str:
        try:
            generation_config: dict = {"temperature": 0.1 if json_mode else 0.2}
            if json_mode:
                generation_config["response_mime_type"] = "application/json"
            with self._lock:
                response = self._gemini.generate_content(
                    f"{system_prompt}\n\n{user_prompt}",
                    generation_config=generation_config,
                )
            content = (response.text or "").strip()
            if not content:
                raise LLMError("Gemini returned an empty response.")
            return content
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc
