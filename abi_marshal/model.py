"""ABI type model.

Types are plain identity-keyed Python objects.  Records are built
incrementally with ``add_field`` so self-referential graphs can be
constructed; ``layout()`` is responsible for rejecting actual cycles
(a record containing itself *by value*) with ``CycleError``.

Fixed ABI contract (see README):
  * little-endian everywhere
  * pointers: 8 bytes, unsigned, integer class
  * integers: 1/2/4/8 bytes, signed or unsigned
  * floats:   8 bytes (IEEE-754 binary64)
  * records:  natural alignment, alignment capped at 8
  * arrays:   explicit length, elements contiguous at full element stride
"""

from .errors import TypeDefinitionError

POINTER_SIZE = 8
MAX_ALIGN = 8
#: Maximum element count of an ArrayType (interface bound, not a target).
MAX_ARRAY_COUNT = 1 << 20


class Type:
    """Base class of all ABI types.  Identity equality/hash."""

    __slots__ = ()


def _check_name(name, what):
    if not isinstance(name, str) or not name:
        raise TypeDefinitionError(f"{what} name must be a non-empty string, got {name!r}")
    return name


class IntType(Type):
    """Integer of 1/2/4/8 bytes, signed or unsigned."""

    __slots__ = ("name", "size", "signed")

    def __init__(self, name, size, signed):
        if type(size) is not int or size not in (1, 2, 4, 8):
            raise TypeDefinitionError(f"integer size must be 1, 2, 4 or 8 bytes, got {size!r}")
        if type(signed) is not bool:
            raise TypeDefinitionError(f"signed must be a bool, got {signed!r}")
        self.name = _check_name(name, "integer type")
        self.size = size
        self.signed = signed

    @property
    def bits(self):
        return self.size * 8

    @property
    def min_value(self):
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def max_value(self):
        return (1 << (self.bits - (1 if self.signed else 0))) - 1

    def __repr__(self):
        return self.name


class FloatType(Type):
    """8-byte IEEE-754 binary64 float."""

    __slots__ = ()
    size = 8

    def __repr__(self):
        return "f64"


class PtrType(Type):
    """8-byte opaque pointer, transported as an unsigned 64-bit integer."""

    __slots__ = ()
    size = POINTER_SIZE

    def __repr__(self):
        return "ptr"


class RecordType(Type):
    """Ordered collection of named fields, naturally aligned."""

    __slots__ = ("name", "fields")

    def __init__(self, name, fields=()):
        self.name = _check_name(name, "record type")
        self.fields = []
        for fname, ftype in fields:
            self.add_field(fname, ftype)

    def add_field(self, fname, ftype):
        _check_name(fname, "field")
        if not isinstance(ftype, Type):
            raise TypeDefinitionError(
                f"field {fname!r} of record {self.name!r} is not an ABI type: {ftype!r}"
            )
        if any(n == fname for n, _ in self.fields):
            raise TypeDefinitionError(f"duplicate field {fname!r} in record {self.name!r}")
        self.fields.append((fname, ftype))
        return self

    def __repr__(self):
        return f"record {self.name}"


class ArrayType(Type):
    """Explicit-length sequence of ``count`` elements of one type."""

    __slots__ = ("element", "count")

    def __init__(self, element, count):
        if not isinstance(element, Type):
            raise TypeDefinitionError(f"array element is not an ABI type: {element!r}")
        if type(count) is not int:  # bool is not an int here
            raise TypeDefinitionError(f"array count must be an int, got {count!r}")
        if not 0 <= count <= MAX_ARRAY_COUNT:
            raise TypeDefinitionError(
                f"array count {count} out of range [0, {MAX_ARRAY_COUNT}]"
            )
        self.element = element
        self.count = count

    def __repr__(self):
        return f"{self.element!r}[{self.count}]"


# Canonical scalar singletons.
I8 = IntType("i8", 1, True)
I16 = IntType("i16", 2, True)
I32 = IntType("i32", 4, True)
I64 = IntType("i64", 8, True)
U8 = IntType("u8", 1, False)
U16 = IntType("u16", 2, False)
U32 = IntType("u32", 4, False)
U64 = IntType("u64", 8, False)
F64 = FloatType()
PTR = PtrType()
