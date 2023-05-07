import os
import copy
import json
import logging

import pandas as pd

from tqdm import tqdm
from typing import Dict, List
from pathlib import Path
from transformers import GPT2TokenizerFast

from business_ocr import Client, OUTPUT_FORMAT_TXT

from src.entities import ResultEntity


class Generator:
    def __init__(self, config: Dict, secrets: Dict, models: List, prompts: List) -> None:
        self.config = config
        self.secrets = secrets

        self.models = models
        self.prompts = prompts

        self.__setup()

    def start(self):
        for model in self.models:
            for prompt in self.prompts:
                self.accuracy_table = pd.DataFrame(columns=self.columns)
                with tqdm(total=self.total_files, desc=f"{model.modelname}, {prompt.name}: Processing files", unit="file") as pbar:
                    for path_file in self.path_docs.iterdir():
                        extracted_text = self.__create_or_load_bocr(path_file)

                        input = prompt.prompt + "\n\n" + extracted_text + "\n\n" + prompt.base
                        predictions = model.predict(input)

                        target = self.__load_target(path_file.stem)
                        accuracies = self.__calculate_accuracies(
                            predictions, target)

                        self.__update_results(
                            model.modelname, prompt.name, input, path_file.stem, predictions, accuracies)

                        pbar.update(1)

                self.__iteration_stop(model.modelname, prompt.name)

    def __calculate_accuracies(self, predictions, target) -> List:
        return target.compare(predictions)

    def __load_target(self, filename):
        with open((self.path_golden / filename).with_suffix(".json"), encoding="utf-8") as f:
            target_json = json.load(f)

        target = ResultEntity(
            subject=target_json["subject"],
            sender=target_json["sender"],
            persons=target_json["persons"],
            companies=target_json["companies"],
            dates=target_json["dates"],
            action=target_json["action"],
            deadline=target_json["deadline"],
            priority=target_json["priority"],
            country=target_json["country"],
            currency=target_json["currency"],
            language=target_json["language"]
        )

        return target

    def __update_results(self, modelname, promptname, input, filename, predictions, accuracies):
        path = (((self.path_results / modelname) / promptname) /
                filename).with_suffix(".json")
        path.parent.parent.mkdir(exist_ok=True)
        path.parent.mkdir(exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(predictions.to_dict(), f, ensure_ascii=False)

        tokens = len(self.tokenizer.encode(str(input))) + \
            len(self.tokenizer.encode(str(predictions)))

        if (predictions.error_message):
            print(f"Error Code: {predictions.error_message}")
            predictions = f"{predictions.error_message}\n{predictions.error_prediction}"

        extended_infos = [filename, predictions, tokens, -1]

        self.accuracy_table.loc[len(
            self.accuracy_table)] = accuracies + extended_infos

    def __setup(self) -> None:
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        self.CLIENT = Client(
            service_key=self.secrets["BOCR_KEY"], logging_level=logging.ERROR)

        self.EXECUTION_NAME = self.config["execution"]
        self.DATA_BASE_PATH = Path(self.config["paths"]["data"])
        self.PROMPT_BASE_PATH = Path(self.config["paths"]["prompts"])
        self.RESULT_BASE_PATH = Path(self.config["paths"]["results"])

        self.path_docs = self.DATA_BASE_PATH / "docs"
        self.path_golden = self.DATA_BASE_PATH / "golden"

        for path in [self.DATA_BASE_PATH, self.PROMPT_BASE_PATH, self.path_docs, self.path_golden]:
            assert path.exists()

        self.path_ocr = self.DATA_BASE_PATH / "ocr"
        self.path_ocr.mkdir(exist_ok=True)

        self.RESULT_BASE_PATH.mkdir(exist_ok=True)
        self.path_results = self.RESULT_BASE_PATH / self.EXECUTION_NAME
        self.path_results.mkdir(exist_ok=True)

        self.columns = copy.deepcopy(self.config["fields"])
        self.columns.extend(["file", "prediction", "tokens", "acc"])
        self.total_files = sum(1 for _ in self.path_docs.iterdir())

    def __create_or_load_bocr(self, path_file) -> str:
        if (self.path_ocr / path_file.stem).with_suffix(".txt").exists():
            with open((self.path_ocr / path_file.stem).with_suffix(".txt"), "r", encoding="utf-8") as f:
                processed_text = f.read().strip()
        else:
            bocr_result, _ = self.CLIENT.create_job_with_polling(
                file_path=os.path.join(path_file.parent, path_file.name), output_format=OUTPUT_FORMAT_TXT)

            with open((self.path_ocr / path_file.stem).with_suffix(".txt"), "w", encoding="utf-8") as f:
                f.write(bocr_result["result"])

            processed_text = bocr_result["result"]
        return processed_text

    def __iteration_stop(self, modelname, promptname):
        print(
            f"Accumulated accuracies: {self.accuracy_table[self.config['fields']].mean(axis=0).mean():.2f}")
        print(
            f"Accuracies for all values: \n{self.accuracy_table[self.config['fields']].mean(axis=0)}", end="\n\n")

        self.accuracy_table["acc"] = self.accuracy_table[self.config["fields"]].mean(
            axis=1)
        self.accuracy_table.to_excel(
            ((self.path_results / modelname) / promptname) / "scores.xlsx")
