"""Functions that act on Directed Acyclic Graphs (DAG)."""
import nitorch.proba.variables as variables
from nitorch import core


def is_independent(x, y, condition=None):
    """Check that two random variables are (conditionally) independent.

    Parameters
    ----------
    x : RandomVariable
    y : RandomVariable
    condition : RandomVariable, optional

    Returns
    -------
    bool

    Note
    ----
    This function uses the concept of D-separation to determine
    conditional independence.

    References
    ----------
    ..[1] "Chapter 8: Graphical Models",
        Christopher Bishop, in "Pattern Recognition and Machine Learning"
        pp 359-422. Springer (2006)

    """
    if (not isinstance(x, variables.RandomVariable) or
        not isinstance(y, variables.RandomVariable)):
        raise TypeError('Inputs must be random variables')
    if condition is None:
        return _is_disjoint(x, y)
    else:
        if not isinstance(condition, variables.RandomVariable):
            raise TypeError('Condition must be a random variable')
        return _is_cond_independent(x, y, condition)


def _parameters(var):
    """Iterator over direct (random) parameters of a random variable."""
    for parent in var.parameters.values():
        if isinstance(parent, variables.RandomVariable):
            yield parent


def _all_parameters(var):
    """Nested iterator over all (random) parameters of a random variable."""
    if isinstance(var, variables.RandomVariable):
        for parent in var.parameters.values():
            if isinstance(parent, variables.RandomVariable):
                yield parent
                for grandparent in _all_parameters(parent):
                    yield grandparent
    else:
        for subvar in var:
            for param in _all_parameters(subvar):
                yield param


def _is_disjoint(x, y):
    # No paths of the form (x) <- (y) or (y) <- (x)
    return (x not in _all_parameters(y)) and (y not in _all_parameters(x))


def _is_cond_independent(x, y, cond):
    def isin(x, c):
        """Same as `is in` but comparisons use `is` instead of `==`"""
        for z in c:
            if x is z:
                return True
        return False

    if _is_disjoint(x, y):
        return True
    cond = core.pyutils.make_list(cond)

    # climb back from x
    # => paths of the form (x) <- (y)
    parents = _parameters(x)
    while parents:
        kept_parents = []
        for parent in parents:
            if parent not in cond:
                # non-blocked path (for now)
                kept_parents.append(parent)
            # else: blocked path -> discard it
            if parent is y:
                # we found an unblocked path from x to y
                # -> not conditionally independent
                return False
        parents = []
        for parent in kept_parents:
            parents.extend(_parameters(parent))

    # climb back from y
    # => paths of the form (y) <- (x)
    parents = _parameters(y)
    while parents:
        kept_parents = []
        for parent in parents:
            if parent not in cond:
                # non-blocked path (for now)
                kept_parents.append(parent)
            # else: blocked path -> discard it
            if parent is x:
                # we found an unblocked path from y to x
                # -> not conditionally independent
                return False
        parents = []
        for parent in kept_parents:
            parents.extend(_parameters(parent))

    # climb back from cond
    # => paths of the form (x) -> (cond) <- (y)
    if isin(x, _all_parameters(cond)) and isin(y, _all_parameters(cond)):
        # the condition is a children of both variables
        # -> since the condition is observed, they are not independent
        return False

    # we have explored all the ways x and y could be related and
    # found nothing, they are therefore independent
    return True
