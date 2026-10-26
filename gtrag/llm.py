"""LLM backends.

Every backend is a callable:  list of chat messages -> assistant reply (str).
The rest of the pipeline only ever calls `chat(messages)`, so adding a new backend means
adding one class here and one branch in build_chat().

    llm.backend: hf      local Hugging Face model (needs a GPU for 8B-scale models)
    llm.backend: openai  any OpenAI-compatible chat API (OpenAI, vLLM, Together, ...)
"""
from __future__ import annotations

import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gtrag.config import Config


def get_device() -> "torch.device":
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- local model
class HFChat:
    """Local Hugging Face causal LM (e.g. Llama-3-8B-Instruct)."""

    def __init__(self, llm_cfg: Config, device):
        torch.cuda.empty_cache()  # free cached memory before loading the model
        if not llm_cfg.model_path:
            raise ValueError("Set llm.model_path (HF id or local path), e.g. --set llm.model_path=...")
        self.cfg, self.device = llm_cfg, device
        self.last_prompt_tokens = 0  # prompt length of the most recent call (for token accounting)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_cfg.tokenizer_path or llm_cfg.model_path)
        kwargs = {}
        if llm_cfg.torch_dtype:
            kwargs["torch_dtype"] = getattr(torch, llm_cfg.torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(llm_cfg.model_path, **kwargs).to(device)

    def __call__(self, conv_history: list[dict]) -> str:
        inputs = self.tokenizer.apply_chat_template(
            conv_history,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        self.last_prompt_tokens = int(inputs["input_ids"].shape[-1])
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            eos_token_id=terminators,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


# ----------------------------------------------------------------------------- API model
class OpenAIChat:
    """OpenAI-compatible chat-completions API. The API key is read from an environment variable."""

    def __init__(self, llm_cfg: Config):
        import openai  # imported lazily: only needed for this backend

        api = llm_cfg.api
        if not api.model:
            raise ValueError("Set llm.api.model, e.g. --set llm.api.model=<model name>")
        key = os.environ.get(api.api_key_env)
        if not key:
            raise ValueError(f"Set the {api.api_key_env} environment variable (never put keys in config files)")
        self.api, self.max_tokens = api, llm_cfg.max_new_tokens
        self.last_prompt_tokens = 0  # prompt tokens reported by the API for the most recent call
        self.client = openai.OpenAI(api_key=key, base_url=api.base_url)
        # Only transient errors are retried; auth / bad-request errors fail immediately.
        self.retryable = (openai.RateLimitError, openai.APIConnectionError,
                          openai.APITimeoutError, openai.InternalServerError)

    def __call__(self, conv_history: list[dict]) -> str:
        api = self.api
        for attempt in range(api.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=api.model,
                    messages=conv_history,
                    temperature=api.temperature,
                    top_p=api.top_p,
                    **{api.max_tokens_param: self.max_tokens},
                )
                usage = getattr(response, "usage", None)
                self.last_prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                return response.choices[0].message.content or ""
            except self.retryable:
                if attempt == api.max_retries - 1:
                    raise
                time.sleep(api.retry_wait_seconds * (2 ** attempt))  # exponential backoff
        raise RuntimeError("unreachable")


# ----------------------------------------------------------------------------- factory
def build_chat(cfg: Config, device):
    backend = cfg.llm.backend
    if backend == "hf":
        return HFChat(cfg.llm, device)
    if backend == "openai":
        return OpenAIChat(cfg.llm)
    raise ValueError(f"Unknown llm.backend '{backend}' (use 'hf' or 'openai')")
