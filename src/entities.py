import jellyfish

from typing import Dict, List

from dataclasses import dataclass, asdict, field


@dataclass
class ResultEntity:
    subject: str = ""
    sender: str = ""
    persons: List = field(default_factory=list)
    companies: List = field(default_factory=list)
    dates: Dict[str, str] = field(default_factory=dict)
    action: str = ""
    deadline: str = ""
    priority: str = ""
    country: str = ""
    currency: str = ""
    language: str = ""
    error_message: str = ""
    error_prediction: str = ""

    def compare(self, other):
        accuracies = []

        if (other.error_message):
            return [0] * (len(self.__dataclass_fields__.keys()) - 2)

        for field in self.__dataclass_fields__.keys():
            if not field.startswith("error"):
                if type(getattr(self, field)) != type(getattr(other, field)):
                    accuracies.append(0)
                elif field in ["subject", "sender", "action"]:
                    similarity = jellyfish.jaro_winkler(
                        getattr(self, field), getattr(other, field))
                    accuracies.append(similarity)
                elif field == "dates":
                    if not all(isinstance(value, str) for value in getattr(other, field).values()):
                        accuracies.append(0)
                    else:
                        date_keys_similarity = self.__jaccard_similarity(
                            self.dates.keys(), other.dates.keys())
                        date_values_similarity = [
                            jellyfish.jaro_winkler(
                                self.dates[date_key], other.dates[date_key])
                            for date_key in self.dates
                            if date_key in other.dates
                        ]

                        if len(date_values_similarity):
                            summed_date_values_similarity = sum(
                                date_values_similarity) / len(date_values_similarity)
                        else:
                            summed_date_values_similarity = 0

                        avg_date_similarity = (
                            date_keys_similarity + summed_date_values_similarity) / 2
                        accuracies.append(avg_date_similarity)
                elif field in ["persons", "companies"]:
                    if not all(isinstance(element, str) for element in getattr(other, field)):
                        accuracies.append(0)
                    else:
                        jaccard = self.__jaccard_similarity(
                            getattr(self, field), getattr(other, field))
                        accuracies.append(jaccard)
                else:
                    accuracies.append(1 if getattr(self, field)
                                      == getattr(other, field) else 0)
        return accuracies

    def __jaccard_similarity(self, list1, list2):
        set1, set2 = set(list1), set(list2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)

        if (len(union) == 0):
            return 1

        return len(intersection) / len(union)

    def to_dict(self):
        return asdict(self)
