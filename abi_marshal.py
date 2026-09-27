"""abi_marshal — 聚合参数的合成 ABI 编组（离线外部函数接口工程）。

本模块实现且仅实现以下明示 ABI 契约（不宣称兼容任何真实平台 ABI）：

* 小端字节序；
* 指针 8 字节（无符号）；整数 1/2/4/8 字节（有符号或无符号）；浮点 8 字节
  （IEEE-754 binary64）；
* 记录（record）按自然对齐布局，对齐上限 8 字节；空记录 size=0、align=1；
* 数组为显式长度的同类元素序列，元素按完整 size 连续排列；
* 聚合按 8 字节块（eightbyte）分类：块内含任意整数/指针字节则该块为
  INTEGER 类，否则为 FLOAT 类；聚合超过 16 字节整体走栈（MEMORY）；
* 整数寄存器 4 个、浮点寄存器 2 个；一个聚合所需寄存器不足时，整体走
  16 字节对齐的栈，且不消耗任何寄存器；
* 空记录不消耗寄存器也不消耗栈。

公开接口：类型构造（IntType/FloatType/PtrType/ArrayType/RecordType 及常量）、
layout()、classify()、encode()、decode()、参数编组 pack_call()/
unpack_call()，以及返回值编组 ResultBuffer、pack_return()、unpack_return()。

错误语义：所有非法输入抛出 ABIError 子类，并尽量携带可定位的 path
（如 ``arg[2].samples[3]``）。不做静默纠正：整数溢出、bool 冒充整数、
NaN/Infinity、字段缺失/多余、数组长度不符、循环类型一律拒绝。
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    # 常量
    "INTEGER", "FLOAT", "MEMORY",
    "INT_REG_COUNT", "FLOAT_REG_COUNT", "STACK_ALIGN", "MAX_EIGHTBYTE_SIZE",
    "MAX_ARGS", "MAX_ARRAY_COUNT", "MAX_RECORD_FIELDS", "MAX_TYPE_SIZE",
    # 错误
    "ABIError", "InvalidTypeError", "RecursiveTypeError",
    "InvalidValueError", "IntegerOverflowError", "NonFiniteFloatError",
    # 类型
    "Type", "IntType", "FloatType", "PtrType", "ArrayType", "RecordType",
    "INT8", "INT16", "INT32", "INT64",
    "UINT8", "UINT16", "UINT32", "UINT64",
    "DOUBLE", "PTR",
    # 布局 / 分类
    "Layout", "FieldLayout", "layout", "classify",
    # 编组
    "Placement", "CallImage",
    "ReturnImage", "ResultBuffer", "DEFAULT_RESULT_ADDRESS",
    "encode", "decode", "pack_call", "unpack_call",
    "pack_return", "unpack_return",
]

# ---------------------------------------------------------------------------
# ABI 常量与输入上限
# ---------------------------------------------------------------------------

INTEGER = "INTEGER"
FLOAT = "FLOAT"
MEMORY = "MEMORY"

INT_REG_COUNT = 4
FLOAT_REG_COUNT = 2
STACK_ALIGN = 16
MAX_EIGHTBYTE_SIZE = 16  # 聚合超过 16 字节整体走栈

MAX_ARGS = 1024                 # 单次调用最多参数个数
MAX_ARRAY_COUNT = 65536         # 数组最大元素数
MAX_RECORD_FIELDS = 256         # 记录最大字段数
MAX_TYPE_SIZE = 1 << 20         # 任一类型完整布局最大字节数（1 MiB）

_INT_SIZES = (1, 2, 4, 8)


# ---------------------------------------------------------------------------
# 错误层级
# ---------------------------------------------------------------------------

class ABIError(Exception):
    """本模块所有错误的基类。``path`` 给出可定位的出错位置。"""

    def __init__(self, message: str, path: str | None = None):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}" if path else message)


class InvalidTypeError(ABIError):
    """类型定义本身非法（未定义的记录、字段重复、尺寸超限等）。"""


class RecursiveTypeError(InvalidTypeError):
    """记录类型直接/间接包含自身，布局无限大，拒绝。"""


class InvalidValueError(ABIError):
    """值与类型不匹配（类型错误、长度不符、字段缺失/多余等）。"""


class IntegerOverflowError(InvalidValueError):
    """整数/指针值超出目标类型的可表示范围。"""


class NonFiniteFloatError(InvalidValueError):
    """浮点值为 NaN 或 Infinity，拒绝。"""


# ---------------------------------------------------------------------------
# 类型描述
# ---------------------------------------------------------------------------

class Type:
    """所有 ABI 类型描述的基类。"""

    __slots__ = ()

    @property
    def name(self) -> str:  # pragma: no cover - 子类覆盖
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class IntType(Type):
    """1/2/4/8 字节整数。"""

    size: int
    signed: bool

    def __post_init__(self):
        if self.size not in _INT_SIZES:
            raise InvalidTypeError(
                f"integer size must be one of {_INT_SIZES}, got {self.size!r}")
        if not isinstance(self.signed, bool):
            raise InvalidTypeError(f"signed must be bool, got {self.signed!r}")

    @property
    def name(self) -> str:
        return f"{'i' if self.signed else 'u'}{self.size * 8}"

    @property
    def range(self) -> tuple[int, int]:
        bits = self.size * 8
        if self.signed:
            return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
        return (0, (1 << bits) - 1)


@dataclass(frozen=True)
class FloatType(Type):
    """8 字节 IEEE-754 binary64 浮点。"""

    @property
    def name(self) -> str:
        return "f64"


@dataclass(frozen=True)
class PtrType(Type):
    """8 字节无符号指针。"""

    @property
    def name(self) -> str:
        return "ptr"


@dataclass(frozen=True)
class ArrayType(Type):
    """显式长度的同类元素序列；元素按完整 size 连续排列。"""

    elem: Type
    count: int

    def __post_init__(self):
        _check_type(self.elem)
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise InvalidTypeError(
                f"array count must be an int, got {self.count!r}")
        if not 0 <= self.count <= MAX_ARRAY_COUNT:
            raise InvalidTypeError(
                f"array count {self.count} out of range [0, {MAX_ARRAY_COUNT}]")

    @property
    def name(self) -> str:
        return f"{self.elem.name}[{self.count}]"


class RecordType(Type):
    """命名字段记录。支持先建壳后 ``define()``，以便表达（并被拒绝的）循环。"""

    __slots__ = ("_name", "_fields")

    def __init__(self, name: str):
        if not isinstance(name, str) or not name:
            raise InvalidTypeError(f"record name must be a non-empty str, got {name!r}")
        self._name = name
        self._fields: tuple[tuple[str, Type], ...] | None = None

    @property
    def name(self) -> str:
        return self._name

    def define(self, fields) -> None:
        """一次性定义字段序列 ``[(field_name, Type), ...]``。只能定义一次。"""
        if self._fields is not None:
            raise InvalidTypeError(f"record {self._name!r} is already defined")
        items = tuple(fields)
        if len(items) > MAX_RECORD_FIELDS:
            raise InvalidTypeError(
                f"record {self._name!r} has {len(items)} fields, "
                f"limit is {MAX_RECORD_FIELDS}")
        seen: set[str] = set()
        for item in items:
            if not (isinstance(item, tuple) and len(item) == 2):
                raise InvalidTypeError(
                    f"record {self._name!r}: field must be a (name, Type) pair, "
                    f"got {item!r}")
            fname, ftype = item
            if not isinstance(fname, str) or not fname:
                raise InvalidTypeError(
                    f"record {self._name!r}: field name must be a non-empty str, "
                    f"got {fname!r}")
            if fname in seen:
                raise InvalidTypeError(
                    f"record {self._name!r}: duplicate field name {fname!r}")
            seen.add(fname)
            _check_type(ftype)
        self._fields = items

    @property
    def fields(self) -> tuple[tuple[str, Type], ...]:
        if self._fields is None:
            raise InvalidTypeError(f"record {self._name!r} is not defined yet")
        return self._fields


INT8 = IntType(1, True)
INT16 = IntType(2, True)
INT32 = IntType(4, True)
INT64 = IntType(8, True)
UINT8 = IntType(1, False)
UINT16 = IntType(2, False)
UINT32 = IntType(4, False)
UINT64 = IntType(8, False)
DOUBLE = FloatType()
PTR = PtrType()


def _check_type(t) -> None:
    if not isinstance(t, Type):
        raise InvalidTypeError(f"expected an ABI Type, got {t!r}")


def _align_up(offset: int, align: int) -> int:
    return (offset + align - 1) // align * align


# ---------------------------------------------------------------------------
# 布局
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldLayout:
    """记录中一个字段的布局。"""

    name: str
    offset: int
    type: Type
    layout: "Layout"


@dataclass(frozen=True)
class Layout:
    """一个类型的完整布局：size、align，以及（记录的）逐字段布局。"""

    type: Type
    size: int
    align: int
    fields: tuple[FieldLayout, ...] = ()


def layout(t: Type, _visiting: frozenset[int] | None = None) -> Layout:
    """计算类型 ``t`` 的字节布局。循环记录类型抛 RecursiveTypeError。"""
    _check_type(t)
    if _visiting is None:
        _visiting = frozenset()

    if isinstance(t, IntType):
        return Layout(t, t.size, t.size)
    if isinstance(t, FloatType):
        return Layout(t, 8, 8)
    if isinstance(t, PtrType):
        return Layout(t, 8, 8)

    if isinstance(t, ArrayType):
        elem_layout = layout(t.elem, _visiting)
        size = elem_layout.size * t.count
        if size > MAX_TYPE_SIZE:
            raise InvalidTypeError(
                f"type size {size} exceeds limit {MAX_TYPE_SIZE}")
        align = elem_layout.align if t.count > 0 else 1
        return Layout(t, size, align)

    if isinstance(t, RecordType):
        if id(t) in _visiting:
            raise RecursiveTypeError(
                f"recursive record type {t.name!r} has infinite layout")
        fields = t.fields  # 未定义时抛 InvalidTypeError
        visiting = _visiting | {id(t)}
        offset = 0
        align = 1
        field_layouts: list[FieldLayout] = []
        for fname, ftype in fields:
            flay = layout(ftype, visiting)
            align = max(align, flay.align)
            offset = _align_up(offset, flay.align)
            field_layouts.append(FieldLayout(fname, offset, ftype, flay))
            offset += flay.size
        size = _align_up(offset, align)
        if size > MAX_TYPE_SIZE:
            raise InvalidTypeError(
                f"type size {size} exceeds limit {MAX_TYPE_SIZE}")
        return Layout(t, size, align, tuple(field_layouts))

    raise InvalidTypeError(f"unsupported type {t!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# 聚合分类（按 8 字节块）
# ---------------------------------------------------------------------------

def _spans(t: Type, lay: Layout, base: int, out: list[tuple[int, int, str]]) -> None:
    """把类型的所有标量叶子展开为 (偏移, 长度, 类) 区间。"""
    if isinstance(t, (IntType, PtrType)):
        out.append((base, lay.size, INTEGER))
    elif isinstance(t, FloatType):
        out.append((base, lay.size, FLOAT))
    elif isinstance(t, ArrayType):
        elem_layout = layout(t.elem)
        step = elem_layout.size
        for i in range(t.count):
            _spans(t.elem, elem_layout, base + i * step, out)
    elif isinstance(t, RecordType):
        for f in lay.fields:
            _spans(f.type, f.layout, base + f.offset, out)
    else:  # pragma: no cover
        raise InvalidTypeError(f"unsupported type {t!r}")


def classify(t: Type) -> tuple[str, ...]:
    """按 8 字节块分类聚合。

    返回逐块的 ``INTEGER``/``FLOAT`` 元组；size 为 0 返回空元组；
    超过 16 字节返回 ``(MEMORY,)``。块内含任意整数/指针字节即为 INTEGER。
    """
    lay = layout(t)
    if lay.size == 0:
        return ()
    if lay.size > MAX_EIGHTBYTE_SIZE:
        return (MEMORY,)
    spans: list[tuple[int, int, str]] = []
    _spans(t, lay, 0, spans)
    classes: list[str] = []
    for chunk_start in range(0, lay.size, 8):
        cls = FLOAT
        for off, size, c in spans:
            if c == INTEGER and off < chunk_start + 8 and off + size > chunk_start:
                cls = INTEGER
                break
        classes.append(cls)
    return tuple(classes)


# ---------------------------------------------------------------------------
# 值编码 / 解码（类型化字节序列）
# ---------------------------------------------------------------------------

def _encode_int(value, lay: Layout, signed: bool, path: str) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidValueError(
            f"expected an int (bool is not accepted), got {value!r}", path)
    lo, hi = lay.type.range
    if not lo <= value <= hi:
        raise IntegerOverflowError(
            f"value {value} out of range [{lo}, {hi}] for {lay.type.name}", path)
    return value.to_bytes(lay.size, "little", signed=signed)


def encode(t: Type, value, lay: Layout | None = None, path: str = "value") -> bytes:
    """把 Python 值按类型 ``t`` 的布局编码为小端字节串（长度恰为 size）。"""
    _check_type(t)
    if lay is None:
        lay = layout(t)

    if isinstance(t, IntType):
        return _encode_int(value, lay, t.signed, path)

    if isinstance(t, PtrType):
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidValueError(
                f"expected an int address (bool is not accepted), got {value!r}",
                path)
        if not 0 <= value <= (1 << 64) - 1:
            raise IntegerOverflowError(
                f"pointer value {value} out of range [0, 2**64 - 1]", path)
        return value.to_bytes(8, "little")

    if isinstance(t, FloatType):
        # 只接受 float：接受 int 会在 |v| >= 2**53 时静默丢精度，拒绝。
        if isinstance(value, bool) or not isinstance(value, float):
            raise InvalidValueError(
                f"expected a float, got {value!r}", path)
        if not math.isfinite(value):
            raise NonFiniteFloatError(
                f"non-finite float {value!r} is rejected", path)
        return struct.pack("<d", value)

    if isinstance(t, ArrayType):
        if not isinstance(value, (list, tuple)):
            raise InvalidValueError(
                f"expected a list/tuple of {t.count} elements, got {value!r}",
                path)
        if len(value) != t.count:
            raise InvalidValueError(
                f"array length {len(value)} != declared count {t.count}", path)
        elem_layout = layout(t.elem)
        return b"".join(
            encode(t.elem, v, elem_layout, f"{path}[{i}]")
            for i, v in enumerate(value)
        )

    if isinstance(t, RecordType):
        if not isinstance(value, Mapping):
            raise InvalidValueError(
                f"expected a mapping of record fields, got {value!r}", path)
        expected = {f.name for f in lay.fields}
        actual = set(value.keys())
        missing = expected - actual
        extra = actual - expected
        if missing:
            raise InvalidValueError(
                f"record {t.name!r} missing fields {sorted(missing)}", path)
        if extra:
            raise InvalidValueError(
                f"record {t.name!r} has unexpected fields {sorted(extra)}", path)
        buf = bytearray(lay.size)
        for f in lay.fields:
            chunk = encode(f.type, value[f.name], f.layout, f"{path}.{f.name}")
            buf[f.offset:f.offset + f.layout.size] = chunk
        return bytes(buf)

    raise InvalidTypeError(f"unsupported type {t!r}")  # pragma: no cover


def decode(t: Type, data: bytes, lay: Layout | None = None, path: str = "value"):
    """把 ``encode`` 的产物按类型 ``t`` 解码回 Python 值。"""
    _check_type(t)
    if lay is None:
        lay = layout(t)
    if not isinstance(data, (bytes, bytearray)):
        raise InvalidValueError(f"expected bytes, got {data!r}", path)
    if len(data) != lay.size:
        raise InvalidValueError(
            f"byte length {len(data)} != layout size {lay.size} for {t.name}",
            path)

    if isinstance(t, IntType):
        return int.from_bytes(data, "little", signed=t.signed)
    if isinstance(t, PtrType):
        return int.from_bytes(data, "little")
    if isinstance(t, FloatType):
        return struct.unpack("<d", data)[0]
    if isinstance(t, ArrayType):
        elem_layout = layout(t.elem)
        step = elem_layout.size
        return [
            decode(t.elem, data[i * step:(i + 1) * step], elem_layout,
                   f"{path}[{i}]")
            for i in range(t.count)
        ]
    if isinstance(t, RecordType):
        return {
            f.name: decode(f.type, data[f.offset:f.offset + f.layout.size],
                           f.layout, f"{path}.{f.name}")
            for f in lay.fields
        }
    raise InvalidTypeError(f"unsupported type {t!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# 调用编组：寄存器分配 + 栈映像
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Placement:
    """一个实参的安放结果。

    kind: ``"registers"`` / ``"stack"`` / ``"none"``（空聚合，不占任何资源）。
    """

    kind: str
    eightbytes: tuple[str, ...]
    int_regs: tuple[int, ...] = ()
    float_regs: tuple[int, ...] = ()
    stack_offset: int | None = None
    size: int = 0


@dataclass(frozen=True)
class CallImage:
    """一次调用的完整编组映像。

    int_regs: 长度 4，未用槽位为 None，已用槽位为该 8 字节块的小端整数值。
    float_regs: 长度 2，未用槽位为 None，已用槽位为该 8 字节块解释的 double。
    stack: 16 字节对齐的栈映像字节串。
    placements: 与实参一一对应的安放结果。
    """

    int_regs: tuple
    float_regs: tuple
    stack: bytes
    placements: tuple[Placement, ...]
    sret_pointer: Placement | None = None
    result_address: int | None = None


def pack_call(args, return_type: Type | None = None,
              result_buffer: "ResultBuffer | None" = None) -> CallImage:
    """把 ``[(Type, value), ...]`` 编组为寄存器 + 栈映像。

    逐参数：size 为 0 的聚合不占资源；超过 16 字节走栈；否则按块申请
    寄存器，寄存器不足时整体走 16 字节对齐的栈且不消耗寄存器。
    任一参数非法时整体抛错，不返回部分结果。

    若提供 ``return_type`` 且返回类型超过 16 字节（``MEMORY`` 类），调用方
    必须同时提供接收大结果的 ``result_buffer``；其地址作为隐藏的第一个
    指针参数，先于所有显式参数参与同一套寄存器/栈分配，因此后续参数的
    寄存器与栈位置会相应顺延。空返回类型不需要结果空间。
    """
    if not isinstance(args, (list, tuple)):
        raise InvalidValueError(
            f"args must be a list/tuple of (Type, value) pairs, got {args!r}")
    if len(args) > MAX_ARGS:
        raise InvalidValueError(
            f"too many arguments: {len(args)} > limit {MAX_ARGS}")

    sret_t = None
    sret_address = None
    if return_type is not None:
        _check_type(return_type)
        ret_lay = layout(return_type)
        ret_cls = classify(return_type)
        if ret_cls == (MEMORY,):
            if not isinstance(result_buffer, ResultBuffer):
                raise InvalidValueError(
                    "MEMORY-class return type requires a caller-provided "
                    f"ResultBuffer for {return_type.name}", "return")
            result_buffer.check_type(return_type, ret_lay)
            sret_t = return_type
            sret_address = result_buffer.address
        elif result_buffer is not None:
            raise InvalidValueError(
                "result_buffer is only valid for a MEMORY-class return type; "
                f"{return_type.name} is returned in registers", "return")

    int_regs: list = [None] * INT_REG_COUNT
    float_regs: list = [None] * FLOAT_REG_COUNT
    stack = bytearray()
    placements: list[Placement] = []
    next_int = 0
    next_float = 0

    def to_stack(data: bytes, cls: tuple[str, ...], size: int) -> Placement:
        off = _align_up(len(stack), STACK_ALIGN)
        stack.extend(b"\x00" * (off - len(stack)))
        stack.extend(data)
        return Placement("stack", cls, (), (), off, size)

    visible_items: list[tuple[str, Type, object]] = []
    for i, item in enumerate(args):
        if not (isinstance(item, tuple) and len(item) == 2):
            raise InvalidValueError(
                f"each argument must be a (Type, value) pair, got {item!r}",
                f"arg[{i}]")
        visible_items.append((f"arg[{i}]", item[0], item[1]))

    items: list[tuple[str, Type, object]] = []
    if sret_t is not None:
        items.append(("return.address", PTR, sret_address))
    items.extend(visible_items)

    sret_pointer = None
    for seq, (item_path, t, value) in enumerate(items):
        try:
            _check_type(t)
            lay = layout(t)
        except ABIError as exc:
            raise type(exc)(exc.message, item_path) from None
        data = encode(t, value, lay, path=item_path)
        cls = classify(t)

        if not cls:
            placements.append(Placement("none", (), (), (), None, 0))
            continue
        if cls == (MEMORY,):
            placements.append(to_stack(data, cls, lay.size))
            continue

        need_int = cls.count(INTEGER)
        need_float = cls.count(FLOAT)
        if (next_int + need_int <= INT_REG_COUNT
                and next_float + need_float <= FLOAT_REG_COUNT):
            iregs: list[int] = []
            fregs: list[int] = []
            for k, c in enumerate(cls):
                chunk = data[8 * k:8 * k + 8].ljust(8, b"\x00")
                if c == INTEGER:
                    int_regs[next_int] = int.from_bytes(chunk, "little")
                    iregs.append(next_int)
                    next_int += 1
                else:
                    float_regs[next_float] = struct.unpack("<d", chunk)[0]
                    fregs.append(next_float)
                    next_float += 1
            placements.append(
                Placement("registers", cls, tuple(iregs), tuple(fregs),
                          None, lay.size))
        else:
            # 寄存器不足：整体走栈，不消耗任何寄存器。
            placements.append(to_stack(data, cls, lay.size))

        if sret_t is not None and seq == 0:
            sret_pointer = placements[-1]

    stack.extend(b"\x00" * (-len(stack) % STACK_ALIGN))
    visible_placements = placements[1:] if sret_t is not None else placements
    return CallImage(tuple(int_regs), tuple(float_regs), bytes(stack),
                     tuple(visible_placements), sret_pointer, sret_address)


def unpack_call(types, image: CallImage) -> list:
    """按 ``types`` 从 ``image`` 重建实参值列表（``pack_call`` 的逆）。"""
    if not isinstance(image, CallImage):
        raise InvalidValueError(f"expected a CallImage, got {image!r}")
    if not isinstance(types, (list, tuple)):
        raise InvalidValueError(
            f"types must be a list/tuple of Type, got {types!r}")
    if len(types) != len(image.placements):
        raise InvalidValueError(
            f"{len(types)} types but image has {len(image.placements)} "
            "placements")

    values = []
    for i, (t, pl) in enumerate(zip(types, image.placements)):
        _check_type(t)
        lay = layout(t)
        path = f"arg[{i}]"
        if pl.kind == "none":
            data = b""
        elif pl.kind == "registers":
            chunks: list[bytes] = []
            int_iter = iter(pl.int_regs)
            float_iter = iter(pl.float_regs)
            for c in pl.eightbytes:
                if c == INTEGER:
                    v = image.int_regs[next(int_iter)]
                    chunks.append(v.to_bytes(8, "little"))
                else:
                    chunks.append(struct.pack("<d", image.float_regs[next(float_iter)]))
            data = b"".join(chunks)[:lay.size]
        elif pl.kind == "stack":
            data = image.stack[pl.stack_offset:pl.stack_offset + pl.size]
            if len(data) != pl.size:
                raise InvalidValueError(
                    f"stack image truncated: need {pl.size} bytes at offset "
                    f"{pl.stack_offset}, have {len(data)}", path)
        else:
            raise InvalidValueError(f"unknown placement kind {pl.kind!r}", path)
        values.append(decode(t, data, lay, path))
    return values


# ---------------------------------------------------------------------------
# 返回值编组：寄存器结果 + 调用方提供的大结果空间
# ---------------------------------------------------------------------------

DEFAULT_RESULT_ADDRESS = 0x7F0000001000


class ResultBuffer:
    """调用方为 MEMORY 类返回值提供的合成接收空间（不访问真实内存）。

    地址必须为非负 64 位整数，并按 16 字节对齐；容量严格等于返回类型的
    完整布局字节数，包含嵌套记录/数组之间的间隙与尾部填充。
    """

    __slots__ = ("_type", "_layout", "_address", "_data")

    def __init__(self, t: Type, address: int = DEFAULT_RESULT_ADDRESS):
        _check_type(t)
        lay = layout(t)
        if classify(t) != (MEMORY,):
            raise InvalidTypeError(
                f"result buffer is only for MEMORY-class types, got {t.name} "
                f"(size={lay.size})")
        if isinstance(address, bool) or not isinstance(address, int):
            raise InvalidValueError(
                f"result address must be an int, got {address!r}")
        if not 0 <= address <= (1 << 64) - 1:
            raise IntegerOverflowError(
                f"result address {address} out of range [0, 2**64 - 1]")
        if address % STACK_ALIGN != 0:
            raise InvalidValueError(
                f"result address 0x{address:x} must be {STACK_ALIGN}-byte "
                "aligned")
        self._type = t
        self._layout = lay
        self._address = address
        self._data = b""

    @property
    def type(self) -> Type:
        return self._type

    @property
    def layout(self) -> Layout:
        return self._layout

    @property
    def address(self) -> int:
        return self._address

    @property
    def contents(self) -> bytes:
        """返回已写入的完整布局字节；尚未接收大结果时抛错。"""
        return self._read()

    def check_type(self, t: Type, lay: Layout | None = None) -> None:
        _check_type(t)
        if lay is None:
            lay = layout(t)
        if t != self._type or lay.size != self._layout.size:
            raise InvalidTypeError(
                f"result buffer was created for {self._type.name} "
                f"({self._layout.size} bytes), cannot use it for {t.name} "
                f"({lay.size} bytes)", "return")

    def _write(self, data: bytes) -> None:
        if len(data) != self._layout.size:
            raise InvalidValueError(
                f"result payload length {len(data)} != buffer size "
                f"{self._layout.size}", "return")
        self._data = bytes(data)

    def _read(self) -> bytes:
        if len(self._data) != self._layout.size:
            raise InvalidValueError(
                f"result buffer at 0x{self._address:x} has not received "
                f"{self._layout.size} bytes", "return")
        return self._data


@dataclass(frozen=True)
class ReturnImage:
    """一次返回的完整编组映像。

    kind 为 ``registers`` 时，块按 ``eightbytes`` 分别进入 int/float 寄存器；
    kind 为 ``memory`` 时，地址在 ``int_regs[0]``，完整内容在结果空间中；
    kind 为 ``none`` 时为空记录返回。
    """

    kind: str
    eightbytes: tuple[str, ...]
    int_regs: tuple
    float_regs: tuple
    size: int = 0
    result_address: int | None = None


def pack_return(return_type: Type, value,
                result_buffer: ResultBuffer | None = None) -> ReturnImage:
    """把返回值按与参数相同的 8 字节块规则编组。

    空记录不占寄存器；不超过 16 字节的标量/记录/数组进入寄存器；超过
    16 字节的大结果写入调用方预先提供的 ``ResultBuffer``，并在第一个整数
    寄存器中回传其地址。
    """
    _check_type(return_type)
    lay = layout(return_type)
    data = encode(return_type, value, lay, path="return")
    cls = classify(return_type)

    if not cls:
        if result_buffer is not None:
            raise InvalidValueError(
                "result_buffer is not valid for an empty return type",
                "return")
        return ReturnImage("none", (),
                           (None,) * INT_REG_COUNT,
                           (None,) * FLOAT_REG_COUNT)

    if cls == (MEMORY,):
        if not isinstance(result_buffer, ResultBuffer):
            raise InvalidValueError(
                "MEMORY-class return requires the caller-provided "
                f"ResultBuffer for {return_type.name}", "return")
        result_buffer.check_type(return_type, lay)
        result_buffer._write(data)
        int_regs = [None] * INT_REG_COUNT
        int_regs[0] = result_buffer.address
        return ReturnImage("memory", (MEMORY,), tuple(int_regs),
                           (None,) * FLOAT_REG_COUNT, lay.size,
                           result_buffer.address)

    if result_buffer is not None:
        raise InvalidValueError(
            "result_buffer is only valid for a MEMORY-class return type; "
            f"{return_type.name} is returned in registers", "return")

    int_regs: list = [None] * INT_REG_COUNT
    float_regs: list = [None] * FLOAT_REG_COUNT
    next_int = 0
    next_float = 0
    for k, c in enumerate(cls):
        chunk = data[8 * k:8 * k + 8].ljust(8, b"\x00")
        if c == INTEGER:
            int_regs[next_int] = int.from_bytes(chunk, "little")
            next_int += 1
        else:
            float_regs[next_float] = struct.unpack("<d", chunk)[0]
            next_float += 1
    return ReturnImage("registers", cls, tuple(int_regs), tuple(float_regs),
                       lay.size)


def unpack_return(return_type: Type, image: ReturnImage,
                  result_buffer: ResultBuffer | None = None):
    """从返回映像（必要时连同结果空间）重建 Python 返回值。"""
    _check_type(return_type)
    if not isinstance(image, ReturnImage):
        raise InvalidValueError(f"expected a ReturnImage, got {image!r}")
    lay = layout(return_type)
    cls = classify(return_type)

    if not cls:
        if image.kind != "none":
            raise InvalidValueError(
                f"empty return type expects kind=none, got {image.kind!r}",
                "return")
        return decode(return_type, b"", lay, "return")

    if cls == (MEMORY,):
        if image.kind != "memory":
            raise InvalidValueError(
                f"{return_type.name} expects kind=memory, got {image.kind!r}",
                "return")
        if not isinstance(result_buffer, ResultBuffer):
            raise InvalidValueError(
                "unpacking a MEMORY-class return requires the matching "
                "ResultBuffer", "return")
        result_buffer.check_type(return_type, lay)
        if image.result_address != result_buffer.address:
            raise InvalidValueError(
                f"return address 0x{image.result_address:x} does not match "
                f"buffer address 0x{result_buffer.address:x}", "return")
        if image.int_regs[0] != result_buffer.address:
            raise InvalidValueError(
                "first integer register does not contain the result address",
                "return")
        data = result_buffer._read()
        return decode(return_type, data, lay, "return")

    if image.kind != "registers":
        raise InvalidValueError(
            f"{return_type.name} expects kind=registers, got {image.kind!r}",
            "return")
    if image.eightbytes != cls:
        raise InvalidValueError(
            f"return classes {image.eightbytes} do not match {cls}", "return")
    chunks: list[bytes] = []
    next_int = 0
    next_float = 0
    for c in cls:
        if c == INTEGER:
            value = image.int_regs[next_int]
            next_int += 1
            if not isinstance(value, int) or isinstance(value, bool):
                raise InvalidValueError(
                    f"expected integer return register value, got {value!r}",
                    "return")
            if not 0 <= value <= (1 << 64) - 1:
                raise IntegerOverflowError(
                    f"return register value {value} is not an unsigned 64-bit "
                    "block", "return")
            chunks.append(value.to_bytes(8, "little"))
        else:
            value = image.float_regs[next_float]
            next_float += 1
            if not isinstance(value, float) or isinstance(value, bool):
                raise InvalidValueError(
                    f"expected float return register value, got {value!r}",
                    "return")
            if not math.isfinite(value):
                raise NonFiniteFloatError(
                    f"non-finite return register value {value!r}", "return")
            chunks.append(struct.pack("<d", value))
    data = b"".join(chunks)[:lay.size]
    return decode(return_type, data, lay, "return")
