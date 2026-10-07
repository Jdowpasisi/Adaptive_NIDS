"""C9 adapters behind one interface (see base.py). ADAPTERS maps a name to its class."""


def _registry() -> dict:
    from xnids.adapt.adabn import AdaBNAdapter
    from xnids.adapt.coral import CORALAdapter
    from xnids.adapt.dann import DANNAdapter
    from xnids.adapt.fewshot import FewShotAdapter
    from xnids.adapt.scaling import ScalingAdapter
    from xnids.adapt.tent import TentAdapter

    return {a.name: a for a in (ScalingAdapter, AdaBNAdapter, TentAdapter, CORALAdapter, DANNAdapter, FewShotAdapter)}


def get(name: str, **params):
    reg = _registry()
    if name not in reg:
        raise KeyError(f"unknown adapter {name!r}; known: {sorted(reg)}")
    return reg[name](**params)
