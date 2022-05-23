class TrainingError(Exception):
    pass


class InvalidParameterError(TrainingError):
    pass


class ClassImportError(TrainingError):
    pass


class FunctionImportError(TrainingError):
    pass


class EntityImportError(TrainingError):
    pass
