import json

from pathlib import Path

from src.generator import Generator
from src.models import SAP_Models, Prompt


with open("config.json", "r") as file:
    config = json.load(file)

with open("secrets.json", "r") as file:
    secrets = json.load(file)


def main():
    models = []
    for model in config["models_used"]:
        models.append(
            SAP_Models(
                secrets=secrets,
                modelname=model,
                params=config["models_config"][model]
            )
        )

    prompts = []
    base = None
    for x in range(4):
        with open(Path(config["paths"]["prompts"]) / f"{x}shot.txt", "r", encoding="utf-8") as file:
            prompt = file.read().strip().replace('"', "'")

            if not base:
                base = prompt
                prompt = ""

            prompts.append(Prompt(
                name=f"{x}shot",
                base=base,
                prompt=prompt
            ))

    generator = Generator(config, secrets, models, prompts)
    generator.start()


if __name__ == "__main__":
    main()
