import os
from typing import Optional, Tuple

import torch
from transformers import BertModel, BertTokenizer


class HFModelLoader:
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

    def load(self) -> Tuple[BertTokenizer, BertModel]:
        tokenizer = self._load_tokenizer()
        model = self._load_model()
        model.to(self.device)
        model.eval()
        return tokenizer, model

    def _load_tokenizer(self) -> BertTokenizer:
        try:
            return BertTokenizer.from_pretrained(
                self.model_name,
                local_files_only=True,
                token=self.hf_token,
            )
        except OSError:
            return BertTokenizer.from_pretrained(
                self.model_name,
                local_files_only=False,
                token=self.hf_token,
            )

    def _load_model(self) -> BertModel:
        common_kwargs = {
            "output_attentions": True,
            "output_hidden_states": True,
            "token": self.hf_token,
        }
        try:
            return BertModel.from_pretrained(
                self.model_name,
                local_files_only=True,
                **common_kwargs,
            )
        except OSError:
            return BertModel.from_pretrained(
                self.model_name,
                local_files_only=False,
                **common_kwargs,
            )
