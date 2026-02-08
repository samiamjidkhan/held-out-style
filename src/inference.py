"""Model inference helpers supporting vLLM and HuggingFace backends."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    stop: list[str] | None = None


class InferenceBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        ...

    @abstractmethod
    def generate_batch(self, prompts: list[str], config: GenerationConfig | None = None) -> list[str]:
        ...


class VLLMBackend(InferenceBackend):
    """OpenAI-compatible API client for vLLM servers."""

    def __init__(self, api_base: str, model_name: str, default_config: GenerationConfig | None = None):
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.default_config = default_config or GenerationConfig()

    def _call_api(self, messages: list[dict], config: GenerationConfig) -> str:
        import urllib.request

        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "stream": False,
        }
        if config.stop:
            payload["stop"] = config.stop

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        with urllib.request.urlopen(req, timeout=600) as resp:
            raw = resp.read().decode("utf-8")

        result = json.loads(raw)
        return result["choices"][0]["message"]["content"]

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        config = config or self.default_config
        messages = [{"role": "user", "content": prompt}]
        return self._call_api(messages, config)

    def generate_with_system(self, system: str, prompt: str, config: GenerationConfig | None = None) -> str:
        """Generate with a system prompt."""
        config = config or self.default_config
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self._call_api(messages, config)

    def generate_batch(self, prompts: list[str], config: GenerationConfig | None = None) -> list[str]:
        config = config or self.default_config
        return [self.generate(p, config) for p in prompts]


class HFBackend(InferenceBackend):
    """HuggingFace transformers direct loading backend."""

    def __init__(self, model_name: str, default_config: GenerationConfig | None = None, device: str = "auto"):
        self.model_name = model_name
        self.default_config = default_config or GenerationConfig()
        self._pipeline = None
        self._device = device

    def _load(self):
        if self._pipeline is not None:
            return

        import torch
        from transformers import pipeline

        logger.info(f"Loading model: {self.model_name}")
        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self._device,
        )

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        self._load()
        config = config or self.default_config

        messages = [{"role": "user", "content": prompt}]
        result = self._pipeline(
            messages,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.temperature > 0,
        )
        return result[0]["generated_text"][-1]["content"]

    def generate_batch(self, prompts: list[str], config: GenerationConfig | None = None) -> list[str]:
        return [self.generate(p, config) for p in prompts]


class VLLMLocalBackend(InferenceBackend):
    """Direct vLLM backend for local inference (faster than API for batches)."""

    def __init__(
        self,
        model_name: str,
        default_config: GenerationConfig | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
    ):
        self.model_name = model_name
        self.default_config = default_config or GenerationConfig()
        self._llm = None
        self._tokenizer = None
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

    def shutdown(self):
        """Release GPU memory by destroying the vLLM model."""
        if self._llm is not None:
            logger.info(f"Shutting down vLLM model: {self.model_name}")
            del self._llm
            self._llm = None
            self._tokenizer = None
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def __del__(self):
        self.shutdown()

    def _load(self):
        if self._llm is not None:
            return

        from vllm import LLM
        from transformers import AutoTokenizer

        logger.info(f"Loading vLLM model: {self.model_name}")
        kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }
        if self.max_model_len:
            kwargs["max_model_len"] = self.max_model_len
        # Auto-detect AWQ quantization
        if "awq" in self.model_name.lower():
            kwargs["quantization"] = "awq"
        self._llm = LLM(**kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        return self.generate_batch([prompt], config)[0]

    def generate_with_system(self, system: str, prompt: str, config: GenerationConfig | None = None) -> str:
        """Generate with a system prompt."""
        self._load()
        config = config or self.default_config

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        formatted = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            stop=config.stop,
        )

        outputs = self._llm.generate([formatted], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def generate_batch(self, prompts: list[str], config: GenerationConfig | None = None) -> list[str]:
        self._load()
        config = config or self.default_config

        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            stop=config.stop,
        )

        # Format as chat
        formatted = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            formatted.append(
                self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )

        outputs = self._llm.generate(formatted, sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]

    def generate_batch_with_system(
        self, system: str, prompts: list[str], config: GenerationConfig | None = None
    ) -> list[str]:
        """Generate batch with a shared system prompt."""
        self._load()
        config = config or self.default_config

        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            stop=config.stop,
        )

        formatted = []
        for p in prompts:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": p},
            ]
            formatted.append(
                self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )

        outputs = self._llm.generate(formatted, sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]


def create_backend(cfg: dict[str, Any]) -> InferenceBackend:
    """Create an inference backend from a config dict."""
    backend_type = cfg.get("backend", "vllm")
    default_config = GenerationConfig(
        temperature=cfg.get("temperature", 0.7),
        max_tokens=cfg.get("max_tokens", 2048),
    )

    if backend_type == "vllm":
        return VLLMBackend(
            api_base=cfg["api_base"],
            model_name=cfg["model_name"],
            default_config=default_config,
        )
    elif backend_type == "vllm_local":
        return VLLMLocalBackend(
            model_name=cfg["model_name"],
            default_config=default_config,
            tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
            max_model_len=cfg.get("max_model_len"),
        )
    elif backend_type == "hf":
        return HFBackend(
            model_name=cfg["model_name"],
            default_config=default_config,
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
