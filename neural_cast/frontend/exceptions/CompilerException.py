from neural_cast.frontend.common.common import CompilerLogger

class CompilerException(Exception):
    def __init__(self, message : str):
        CompilerLogger().error(message)
        super().__init__(message)