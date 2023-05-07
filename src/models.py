import re
import ast
import json
import requests

from typing import Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.entities import ResultEntity


@dataclass
class Prompt:
    name: str
    base: str
    prompt: str


class BaseLLM(ABC):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def predict(self):
        pass


# Used for OpenAI GPT-series, GPT-J and BLOOM
class SAP_Models(BaseLLM):
    def __init__(self, secrets: Dict, modelname: str, params: Dict[str, str]) -> None:
        self.secrets = secrets
        self.modelname = modelname
        self.parameters = json.dumps(params).replace('"', r'\"')

        self.setup()

    def setup(self) -> None:
        self.URL = self.secrets["SAP_AI"]["URL"]
        self.HEADERS = self.secrets["SAP_AI"]["HEADERS"]

    def predict(self, input: str) -> ResultEntity:
        escaped_input = json.dumps(input)[1:-1]
        payload = f'{{"message": "{escaped_input}", "model": "{self.modelname}", "parameters": "{self.parameters}"}}'
        payload_bytes = payload.encode("utf8")

        predictions = requests.request(
            "POST", self.URL, headers=self.HEADERS, data=payload_bytes)

        try:
            prediction = predictions.json()["value"]

            if (isinstance(prediction, str)):
                pattern = r'(?s){.*}'
                match = re.search(pattern, prediction)
                prediction = match.group(0)

                try:
                    prediction = json.loads(prediction)
                except Exception:
                    # Unsafer method converting a str to dict. Helps for small JSON string errors
                    prediction = ast.literal_eval(prediction)

            # Checks if the key property is valid and creates instance for all valid key/value pairs
            valid_keys = {key: value for key, value in prediction.items(
            ) if key in ResultEntity.__init__.__code__.co_varnames}
            result_entity = ResultEntity(**valid_keys)

            return result_entity

        except Exception as error:
            if predictions.json() and "value" in predictions.json().keys():
                error_prediction = predictions.json()["value"]
            else:
                error_prediction = "NotAvail"

            error_entity = ResultEntity(
                error_message=str(error),
                error_prediction=str(error_prediction)
            )
            return error_entity
