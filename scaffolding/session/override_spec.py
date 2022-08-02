from collections import UserDict, UserList


def override_spec(old_spec: dict, new_spec: dict) -> dict:
    new_spec = parse_object(new_spec)
    d = override_dict(old_spec, new_spec)
    return remove_meta_data(d)


def remove_meta_data(obj):
    """Recursively traverse the data structure, convert every instance of MetaDict to dict,
    convert every instance of MetaList to list"""
    if isinstance(obj, dict) or isinstance(obj, MetaDict):
        return {k: remove_meta_data(v) for k, v in obj.items()}

    if isinstance(obj, list) or isinstance(obj, MetaList):
        return [remove_meta_data(v) for v in obj]

    return obj


class MetaDict(UserDict):
    pass


class MetaList(UserList):
    pass


def parse_object(obj):
    """Recursively parse deep nested dict or list structure into override specification.

    Resulting object will contain original entries + some metadata at every level
    of depth/nesting (this will be used to control overriding behavior by override_dict
    and override_list functions).

    dict -> MetaDict
    dict with metadata -> MetaDict or MetaList
    list -> MetaList
    values of other types are kept unchanged
    """
    if isinstance(obj, list):
        alist = MetaList([parse_object(item) for item in obj])
        alist.replace_strategy = "replace"
        return alist
    elif isinstance(obj, dict) and "options" not in obj:
        adict = MetaDict({k: parse_object(v) for k, v in obj.items()})
        adict.replace_strategy = "override"
        return adict
    elif isinstance(obj, dict):
        strategy = obj.get("replace_strategy", "replace")
        options = obj["options"]

        items = parse_object(options)
        items.replace_strategy = strategy
        if isinstance(options, list):
            items.override_key = obj["override_key"]
        return items
    else:
        return obj


def override_dict(old_spec: dict, new_spec) -> dict:
    if new_spec.replace_strategy == "replace":
        return dict(new_spec)

    old_spec = dict(old_spec)
    new_spec = dict(new_spec)

    for k, v in new_spec.items():
        old_value = old_spec.get(k)
        if not old_value:
            old_spec[k] = v
        else:
            both_dicts = isinstance(old_value, dict) and isinstance(v, MetaDict)
            both_lists = isinstance(old_value, list) and isinstance(v, MetaList)

            if both_dicts:
                old_spec[k] = override_dict(old_spec[k], v)
            elif both_lists:
                old_spec[k] = override_list(old_value, v)
            else:
                old_spec[k] = v

    return old_spec


def override_list(old_list: list, new_list) -> list:
    if new_list.replace_strategy == 'replace':
        return list(new_list)

    key_set = new_list.override_key
    old_list = list(old_list)
    new_list = list(new_list)

    for new_item in new_list:
        if isinstance(new_item, dict) or isinstance(new_item, MetaDict):
            idx = find_first_dict_index(old_list, new_item, key_set)
            if idx is not None:
                old_list[idx] = override_dict(old_list[idx], new_item)
            else:
                old_list.append(new_item)
        else:
            # there is no way to override, replace the whole list with a new one
            return new_list

    return old_list


def find_first_dict_index(items: list, d: dict, key_set):
    item_indices = [i for i, item in enumerate(items)
                    if dicts_equal(item, d, key_set)]

    if len(item_indices) > 1:
        raise InvalidNumberOfMatchesError()
    return item_indices[0] if len(item_indices) > 0 else None


class InvalidNumberOfMatchesError(Exception):
    pass


def dicts_equal(d1: dict, d2: dict, key_set):
    if not isinstance(d1, dict):
        raise NotDictError(f'Expects d1 to be dictionary. Got {type(d1)}')

    if not (isinstance(d2, dict) or isinstance(d2, MetaDict)):
        raise NotDictError(f'Expects d2 to be dictionary. Got {type(d2)}')

    try:
        for key in key_set:
            if d1[key] != d2[key]:
                return False
    except KeyError:
        return False

    return True


class NotDictError(Exception):
    pass
