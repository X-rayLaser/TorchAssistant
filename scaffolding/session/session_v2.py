from . import parse


class Definition:
    def __init__(self, obj_dict):
        pass

    def parse(self):
        obj = 3
        self._finalize(obj)

    def _finalize(self, obj):
        pass


def parse_dataset(definition):
    name = definition["name"]
    return object()


def build_objects(definitions, mode="configure"):
    for definition in definitions:
        if definition["type"] == "dataset":
            obj = parse_dataset(definition)
