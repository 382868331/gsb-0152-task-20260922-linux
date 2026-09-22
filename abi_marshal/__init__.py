"""abi_marshal: synthetic-ABI marshalling of aggregate call arguments.

Public surface:
  types:    IntType, FloatType, PtrType, RecordType, ArrayType and the
            I8..U64/F64/PTR singletons
  layout:   layout(typ) -> Layout(size, align, fields/element/count)
            classify(typ) -> (INTEGER|FLOAT, ...) | MEMORY | ()
  calls:    plan_call(types) -> CallPlan
            pack_call(types, values) -> CallImage
            unpack_call(types, image) -> list of values
  codecs:   encode_value / decode_value for single values
  errors:   ABIError and subclasses, each carrying a locatable ``path``
"""

from .errors import (
    ABIError,
    ArityError,
    CycleError,
    ImageError,
    IntegerOverflowError,
    NonFiniteFloatError,
    SizeLimitError,
    TypeDefinitionError,
    ValueTypeError,
    format_path,
)
from .layout import (
    FLOAT,
    INTEGER,
    MEMORY,
    MAX_TYPE_SIZE,
    FieldLayout,
    Layout,
    align_up,
    classify,
    layout,
)
from .marshal import (
    FLOAT_REG_COUNT,
    INT_REG_COUNT,
    ArgPlacement,
    CallImage,
    CallPlan,
    decode_value,
    encode_value,
    pack_call,
    plan_call,
    unpack_call,
)
from .model import (
    F64,
    I8,
    I16,
    I32,
    I64,
    PTR,
    U8,
    U16,
    U32,
    U64,
    MAX_ARRAY_COUNT,
    ArrayType,
    FloatType,
    IntType,
    PtrType,
    RecordType,
    Type,
)

__all__ = [
    "ABIError", "ArityError", "CycleError", "ImageError", "IntegerOverflowError",
    "NonFiniteFloatError", "SizeLimitError", "TypeDefinitionError",
    "ValueTypeError", "format_path",
    "FLOAT", "INTEGER", "MEMORY", "MAX_TYPE_SIZE", "FieldLayout", "Layout",
    "align_up", "classify", "layout",
    "FLOAT_REG_COUNT", "INT_REG_COUNT", "ArgPlacement", "CallImage", "CallPlan",
    "decode_value", "encode_value", "pack_call", "plan_call", "unpack_call",
    "F64", "I8", "I16", "I32", "I64", "PTR", "U8", "U16", "U32", "U64",
    "MAX_ARRAY_COUNT", "ArrayType", "FloatType", "IntType", "PtrType",
    "RecordType", "Type",
]
