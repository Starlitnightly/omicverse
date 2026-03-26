# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function, division, absolute_import

from functools import wraps

NoneType = type(None)

PRIMITIVE_TYPES = (bool, float, NoneType, str, int)

def return_primitive(fn):
    """
    Decorator which wraps a single argument function to ignore any
    arguments of primitive type (simply returning them unmodified).
    """
    @wraps(fn)
    def wrapped_fn(x):
        if isinstance(x, PRIMITIVE_TYPES):
            return x
        return fn(x)
    return wrapped_fn

class Serializable(object):
    """
    Base class for all PyEnsembl objects which provides default
    methods such as to_json, from_json, __reduce__, and from_dict

    Relies on the following condition:
         (1) a user-defined to_dict method
         (2) the keys of to_dict() must match the arguments to __init__
    """

    def __str__(self):
        return "%s(%s)" % (
            self.__class__.__name__,
            ", ".join("%s=%s" % (k, v) for (k, v) in self.to_dict().items()))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.to_dict() == other.to_dict()

    def to_dict(self):
        """
        Returns a dictionary which can be used to reconstruct an instance
        of a derived class (typically by matching args to __init__). The values
        of the returned dictionary must be primitive atomic types
        (bool, string, int, float), primitive collections
        (int, list, tuple, set) or instances of Serializable.

        The default implementation is to assume all the arguments to __init__
        have fields of the same name on a serializable object.
        """
        return simple_object_to_dict(self)

    def __hash__(self):
        return hash(tuple(sorted(self.to_dict().items())))

    @classmethod
    def _reconstruct_nested_objects(cls, state_dict):
        """
        Nested serializable objects will be represented as dictionaries so we
        allow manual reconstruction of those objects in this method.

        By default just returns the state dictionary unmodified.
        """
        return state_dict

    # dictionary mapping old keywords to either new names or
    # None if the keyword has been removed from a class
    _SERIALIZABLE_KEYWORD_ALIASES = {}

    @classmethod
    def _update_kwargs(cls, kwargs):
        """
        Rename any old keyword arguments to preserve backwards compatibility
        """
        # check every class in the inheritance chain for its own
        # definition of _KEYWORD_ALIASES
        for klass in cls.mro():
            keyword_rename_dict = getattr(klass, '_SERIALIZABLE_KEYWORD_ALIASES', {})
            for (old_name, new_name) in  keyword_rename_dict.items():
                if old_name in kwargs:
                    old_value = kwargs.pop(old_name)
                    if new_name and new_name not in kwargs:
                        kwargs[new_name] = old_value
        return kwargs

    @classmethod
    def from_dict(cls, state_dict):
        """
        Given a dictionary of flattened fields (result of calling to_dict()),
        returns an instance.
        """
        state_dict = cls._reconstruct_nested_objects(state_dict)
        return cls(**cls._update_kwargs(state_dict))

    def to_json(self):
        """
        Returns a string containing a JSON representation of this Genome.
        """
        return to_json(self)

    @classmethod
    def from_json(cls, json_string):
        """
        Reconstruct an instance from a JSON string.
        """
        return from_json(json_string)

    def write_json_file(self, path):
        """
        Serialize this VariantCollection to a JSON representation and write it
        out to a text file.
        """
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def read_json_file(cls, path):
        """
        Construct a VariantCollection from a JSON file.
        """
        with open(path, 'r') as f:
            json_string = f.read()
        return cls.from_json(json_string)

    def __reduce__(self):
        """
        Overriding this method directs the default pickler to reconstruct
        this object using our from_dict method.
        """

        # I wish I could just return (self.from_dict, (self.to_dict(),) but
        # Python 2 won't pickle the class method from_dict so instead have to
        # use globally defined functions.
        return (from_serializable_repr, (to_serializable_repr(self),))

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helper functions for deconstructing classes, functions, and user-defined
objects into serializable types.
"""
from types import FunctionType, BuiltinFunctionType

import simplejson as json


def init_arg_names(obj):
    """
    Names of arguments to __init__ method of this object's class.
    """
    # doing something wildly hacky by pulling out the arguments to
    # __init__ or __new__ and hoping that they match fields defined on the
    # object
    try:
        init_code = obj.__init__.__func__.__code__
    except AttributeError:
        try:
            init_code = obj.__new__.__func__.__code__
        except AttributeError:
            # if object is a namedtuple then we can return its fields
            # as the required initial args
            if hasattr(obj, "_fields"):
                return obj._fields
            else:
                raise ValueError("Cannot determine args to %s.__init__" % (obj,))

    arg_names = init_code.co_varnames[:init_code.co_argcount]
    # drop self argument
    nonself_arg_names = arg_names[1:]
    return nonself_arg_names

def simple_object_to_dict(self):
    return {name: getattr(self, name) for name in init_arg_names(self)}

def _lookup_value(module_string, name, _cache={}):
    key = (module_string, name)
    if key in _cache:
        value = _cache[key]
    else:
        module_parts = module_string.split(".")
        value = None
        for i in range(1, len(module_parts) + 1):
            try:
                # try importing successively longer chains of
                # sub-modules but break when we hit something that's
                # not a module but actually data
                qualified_name = ".".join(module_parts[:i])
                value = __import__(
                    qualified_name,
                    fromlist=module_parts[:i - 1])
            except ImportError:
                break

        if value is None:
            raise ImportError(module_parts[0])
        # once we've imported as much as we can, continue with getattr
        # lookups
        for attribute_name in module_parts[i:] + name.split("."):
            value = getattr(value, attribute_name)
        _cache[key] = value
    return value


def class_from_serializable_representation(class_repr):
    """
    Given the name of a module and a class it contains, imports that module
    and gets the class object from it.
    """
    return _lookup_value(class_repr["__module__"], class_repr["__name__"])

def get_module_name(obj):
    module_name = obj.__module__
    return module_name

def class_to_serializable_representation(cls):
    """
    Given a class, return two strings:
        - fully qualified import path for its module
        - name of the class

    The class can be reconstructed from these two strings by calling
    class_from_serializable_representation.
    """
    return {"__module__": get_module_name(cls), "__name__": cls.__name__}

def function_from_serializable_representation(fn_repr):
    """
    Given the name of a module and a function it contains, imports that module
    and gets the class object from it.
    """
    return _lookup_value(fn_repr["__module__"], fn_repr["__name__"])

def function_to_serializable_representation(fn):
    """
    Converts a Python function into a serializable representation. Does not
    currently work for methods or functions with closure data.
    """
    if type(fn) not in (FunctionType, BuiltinFunctionType):
        raise ValueError(
            "Can't serialize %s : %s, must be globally defined function" % (
                fn, type(fn),))

    if hasattr(fn, "__closure__") and fn.__closure__ is not None:
        raise ValueError("No serializable representation for closure %s" % (fn,))

    return {"__module__": get_module_name(fn), "__name__": fn.__name__}

SERIALIZED_DICTIONARY_KEYS_FIELD = "__serialized_keys__"
SERIALIZED_DICTIONARY_KEYS_ELEMENT_PREFIX = (
    SERIALIZED_DICTIONARY_KEYS_FIELD + "element_")

def index_to_serialized_key_name(index):
    return "%s%d" % (SERIALIZED_DICTIONARY_KEYS_ELEMENT_PREFIX, index)

def parse_serialized_keys_index(name):
    """
    Given a field name such as __serialized_keys__element_10 returns the integer 10
    but returns None for other strings.
    """
    if name.startswith(SERIALIZED_DICTIONARY_KEYS_ELEMENT_PREFIX):
        try:
            return int(name[len(SERIALIZED_DICTIONARY_KEYS_ELEMENT_PREFIX):])
        except:
            pass
    return None

def dict_to_serializable_repr(x):
    """
    Recursively convert values of dictionary to serializable representations.
    Convert non-string keys to JSON representations and replace them in the
    dictionary with indices of unique JSON strings (e.g. __1, __2, etc..).
    """
    # list of JSON representations of hashable objects which were
    # used as keys in this dictionary
    serialized_key_list = []
    serialized_keys_to_names = {}
    # use the class of x rather just dict since we might want to convert
    # derived classes such as OrderedDict
    result = type(x)()
    for (k, v) in x.items():
        if not isinstance(k, str):
            # JSON does not support using complex types such as tuples
            # or user-defined objects with implementations of __hash__ as
            # keys in a dictionary so we must keep the serialized
            # representations of such values in a list and refer to indices
            # in that list
            serialized_key_repr = to_json(k)
            if serialized_key_repr in serialized_keys_to_names:
                k = serialized_keys_to_names[serialized_key_repr]
            else:
                k = index_to_serialized_key_name(len(serialized_key_list))
                serialized_keys_to_names[serialized_key_repr] = k
                serialized_key_list.append(serialized_key_repr)
        result[k] = to_serializable_repr(v)
    if len(serialized_key_list) > 0:
        # only include this list of serialized keys if we had any non-string
        # keys
        result[SERIALIZED_DICTIONARY_KEYS_FIELD] = serialized_key_list
    return result

def from_serializable_dict(x):
    """
    Reconstruct a dictionary by recursively reconstructing all its keys and
    values.

    This is the most hackish part since we rely on key names such as
    __name__, __class__, __module__ as metadata about how to reconstruct
    an object.

    TODO:
        It would be cleaner to always wrap each object in a layer of type
        metadata and then have an inner dictionary which represents the
        flattened result of to_dict() for user-defined objects.
    """
    if "__name__" in x:
        return _lookup_value(x.pop("__module__"), x.pop("__name__"))

    non_string_key_objects = [
        from_json(serialized_key)
        for serialized_key
        in x.pop(SERIALIZED_DICTIONARY_KEYS_FIELD, [])
    ]
    converted_dict = type(x)()
    for k, v in x.items():
        serialized_key_index = parse_serialized_keys_index(k)
        if serialized_key_index is not None:
            k = non_string_key_objects[serialized_key_index]

        converted_dict[k] = from_serializable_repr(v)
    if "__class__" in converted_dict:
        class_object = converted_dict.pop("__class__")
        if "__value__" in converted_dict:
            return class_object(converted_dict["__value__"])
        elif hasattr(class_object, "from_dict"):
            return class_object.from_dict(converted_dict)
        else:
            return class_object(**converted_dict)
    return converted_dict

def list_to_serializable_repr(x):
    return [to_serializable_repr(element) for element in x]

def to_dict(obj):
    """
    If value wasn't isn't a primitive scalar or collection then it needs to
    either implement to_dict (instances of Serializable) or has member
    data matching each required arg of __init__.
    """
    if isinstance(obj, dict):
        return obj
    elif hasattr(obj, "to_dict"):
        return obj.to_dict()
    try:
        return simple_object_to_dict(obj)
    except:
        raise ValueError(
            "Cannot convert %s : %s to dictionary" % (
                obj, type(obj)))

@return_primitive
def to_serializable_repr(x):
    """
    Convert an instance of Serializable or a primitive collection containing
    such instances into serializable types.
    """
    t = type(x)
    if isinstance(x, list):
        return list_to_serializable_repr(x)
    elif t in (set, tuple):
        return {
            "__class__": class_to_serializable_representation(t),
            "__value__": list_to_serializable_repr(x)
        }
    elif isinstance(x, dict):
        return dict_to_serializable_repr(x)
    elif isinstance(x, (FunctionType, BuiltinFunctionType)):
        return function_to_serializable_representation(x)
    elif type(x) is type:
        return class_to_serializable_representation(x)
    else:
        state_dictionary = to_serializable_repr(to_dict(x))
        state_dictionary["__class__"] = class_to_serializable_representation(
            x.__class__)
        return state_dictionary

@return_primitive
def from_serializable_repr(x):
    t = type(x)
    if isinstance(x, list):
        return t([from_serializable_repr(element) for element in x])
    elif isinstance(x, dict):
        return from_serializable_dict(x)
    else:
        raise TypeError(
            "Cannot convert %s : %s from serializable representation to object" % (
                x, type(x)))

def to_json(x):
    """
    Returns JSON representation of a given Serializable instance or
    other primitive object.
    """
    return json.dumps(to_serializable_repr(x))

def from_json(json_string):
    return from_serializable_repr(json.loads(json_string))

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A type is considered "primitive" if it is a non-collection type which can
be serialized and deserialized successfully.
"""
