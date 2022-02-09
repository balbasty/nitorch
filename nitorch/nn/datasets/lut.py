
def true_name(label, aliases):
    """Convert an alias to its unique name

    Parameters
    ----------
    label : str
    aliases : dict or iterable[(str, list)]

    Returns
    -------
    true_name : str

    """
    if isinstance(aliases, dict):
        aliases = aliases.items()
    for true_name, alias_names in aliases:
        if isinstance(alias_names, str):
            if label.lower() == alias_names.lower():
                return true_name
        else:
            if label in [a.lower() for a in alias_names]:
                return true_name
    return label


def isin(child, parent, relations, aliases=None):
    """Check if a label is under another in a given hierarchy

    Parameters
    ----------
    child : str
    parent : str
    relations : dict or iterable[(str, list)]
    aliases : dict or iterable[(str, list)]

    Returns
    -------
    bool

    """
    if aliases:
        child = true_name(child, aliases)
        parent = true_name(parent, aliases)

    relations = dict(relations)

    def _isin(child, parent):
        if child.lower() == parent.lower():
            return True
        for new_parent, children in relations.items():
            if child.lower() in [c.lower() for c in children]:
                if _isin(new_parent, parent):
                    return True
        return False

    return _isin(child, parent)


def id_to_label(lut):
    """Ensure that the LUT maps (integer) IDs to (string) names"""
    if isinstance(next(iter(lut.keys())), int):
        return lut
    return {v: k for k, v in lut.items()}


def label_to_id(lut):
    """Ensure that the LUT maps (string) names to (integer) IDs"""
    if isinstance(next(iter(lut.keys())), str):
        return lut
    return {v: k for k, v in lut.items()}


def ids_in_group(group, lut, hierarchy):
    """Return IDs of all labels that belong to a group in a hierarchy.

    Parameters
    ----------
    group : str
    lut : dict[int -> str]
    hierarchy : dict[str -> list[str]]

    Returns
    -------
    ids : list[int]

    """
    lut = id_to_label(lut)
    return [k for k, v in lut.items() if isin(v, group, hierarchy)]


def labels_in_group(group, lut, hierarchy):
    """Return names of all labels that belong to a group in a hierarchy.

    Parameters
    ----------
    group : str
    lut : dict[int -> str]
    hierarchy : dict[str -> list[str]]

    Returns
    -------
    ids : list[str]

    """
    lut = id_to_label(lut)
    return [v for v in lut.values() if isin(v, group, hierarchy)]