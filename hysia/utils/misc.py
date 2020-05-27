class _Struct(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [_Struct(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, _Struct(b) if isinstance(b, dict) else b)


def dict_to_object(dictionary: dict):
    return _Struct(dictionary)
