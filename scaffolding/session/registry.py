group_to_loader = {}


def register(group):
    def wrapper(f):
        group_to_loader[group] = f
        return f
    return wrapper
