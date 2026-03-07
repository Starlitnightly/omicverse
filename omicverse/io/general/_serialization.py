import os

from ..._registry import register_function


@register_function(
    aliases=['保存对象', 'save', 'pickle save'],
    category="utils",
    description="Persist Python objects (models, results, intermediate analysis states) for reproducible downstream reuse.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['ov.utils.save(cpdb_results, "data/cpdb/gex_cpdb_test.pkl")'],
    related=['utils.load']
)
def save(file, path):
    """
    Save Python object to file using pickle fallback strategy.

    Parameters
    ----------
    file : Any
        Python object to serialize.
    path : str
        Output file path.
    """
    print("💾 Save Operation:")
    print(f"   Target path: {path}")
    print(f"   Object type: {type(file).__name__}")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        import pickle
        print("   Using: pickle")
        with open(path, 'wb') as f:
            pickle.dump(file, f)
        print("   ✅ Successfully saved!")
    except Exception:
        import cloudpickle
        print("   Pickle failed, switching to: cloudpickle")
        with open(path, 'wb') as f:
            cloudpickle.dump(file, f)
        print("   ✅ Successfully saved using cloudpickle!")
    print("─" * 60)


@register_function(
    aliases=['加载对象', 'load', 'pickle load'],
    category="utils",
    description="Load serialized analysis objects previously saved with ov.utils.save to resume computation or visualization.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['cpdb_results = ov.utils.load("data/cpdb/gex_cpdb_test.pkl")'],
    related=['utils.save']
)
def load(path, backend=None):
    """
    Load serialized Python object from disk.

    Parameters
    ----------
    path : str
        Input file path.
    backend : str or None
        Preferred deserializer backend (``'pickle'`` or ``'cloudpickle'``).
    """
    print("📂 Load Operation:")
    print(f"   Source path: {path}")
    if backend is None:
        try:
            import pickle
            print("   Using: pickle")
            with open(path, 'rb') as f:
                data = pickle.load(f)
            print("   ✅ Successfully loaded!")
            print(f"   Loaded object type: {type(data).__name__}")
            print("─" * 60)
            return data
        except Exception:
            import cloudpickle
            print("   Pickle failed, switching to: cloudpickle")
            with open(path, 'rb') as f:
                data = cloudpickle.load(f)
            print("   ✅ Successfully loaded using cloudpickle!")
            print(f"   Loaded object type: {type(data).__name__}")
            print("─" * 60)
            return data

    if backend == 'pickle':
        import pickle
        print("   Using: pickle")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print("   ✅ Successfully loaded!")
        print(f"   Loaded object type: {type(data).__name__}")
        print("─" * 60)
        return data

    if backend == 'cloudpickle':
        import cloudpickle
        print("   Using: cloudpickle")
        with open(path, 'rb') as f:
            data = cloudpickle.load(f)
        print("   ✅ Successfully loaded!")
        print(f"   Loaded object type: {type(data).__name__}")
        print("─" * 60)
        return data

    raise ValueError(f"Invalid backend: {backend}")
