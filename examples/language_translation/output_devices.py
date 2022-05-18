class Printer:
    """How to present the result"""
    def __call__(self, result_dict):
        for k, v in result_dict.items():
            print(f'{k}:{v}')
