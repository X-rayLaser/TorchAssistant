group_to_loader = {}


def definition(group):
    def wrapper(f):
        group_to_loader[group] = f
        return f
    return wrapper
