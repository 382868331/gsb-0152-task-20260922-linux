"""Error hierarchy for abi_marshal.

Every error carries a ``path`` tuple locating the offending type node or
value, e.g. ``("arg0", "x")`` for field ``x`` of the first argument, or
``("arg2", 3)`` for element 3 of an array argument.  ``str(err)`` renders
the path in a ``arg0.x[3]`` style so failures are locatable without a
debugger.  No input is ever silently corrected: anything outside the
contract raises one of these errors.
"""


def format_path(path):
    """Render a path tuple as ``arg0.x[3]`` style text."""
    if not path:
        return "<root>"
    out = []
    for seg in path:
        if isinstance(seg, int):
            out.append(f"[{seg}]")
        else:
            if out:
                out.append(".")
            out.append(str(seg))
    return "".join(out)


class ABIError(Exception):
    """Base class for all abi_marshal errors."""

    def __init__(self, message, path=()):
        self.message = message
        self.path = tuple(path)
        super().__init__(str(self))

    def __str__(self):
        if self.path:
            return f"{self.message} (at {format_path(self.path)})"
        return self.message


class TypeDefinitionError(ABIError):
    """A type object is malformed (bad scalar size, bad array count, ...)."""


class CycleError(TypeDefinitionError):
    """A record type (transitively) contains itself as a field."""


class SizeLimitError(TypeDefinitionError):
    """A type exceeds the documented size/count limits."""


class ArityError(ABIError):
    """Wrong number of call arguments or array elements."""


class ValueTypeError(ABIError):
    """A Python value has the wrong kind for its ABI type (incl. bool)."""


class IntegerOverflowError(ABIError):
    """An integer/pointer value is outside the range of its ABI type."""


class NonFiniteFloatError(ABIError):
    """A float value is NaN or +/-Infinity, which this ABI rejects."""


class ImageError(ABIError):
    """A CallImage passed to unpack_call is malformed or truncated."""
