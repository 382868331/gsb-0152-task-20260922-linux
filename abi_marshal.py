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

返回值约定（与参数编组配套）：

* 标量及不超过 16 字节的聚合沿用同一块分类，经返回寄存器传递：整数类
  返回寄存器 2 个、浮点类返回寄存器 2 个（至多 2 块，必然放得下）；
* 空记录不占用任何返回寄存器；
* 超过 16 字节的聚合由调用方传入有界返回缓冲区（ReturnBuffer），被调方
  把隐藏指针（缓冲区地址）先占一个整数参数寄存器，再安排原参数——因此
  它可能把后面的整个聚合挤到栈上；缓冲区的大小、对齐和范围在编码任何
  参数之前先验证，失败不写入任何部分结果；
* 所有 padding 写零；读取时 padding 不作为值解释。

公开接口：类型构造（IntType/FloatType/PtrType/ArrayType/RecordType 及常量）、
layout()、classify()、encode()、decode()、pack_call()、unpack_call()、
plan_return()、pack_return()、unpack_return()、simulate_call()、
ReturnBuffer。

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
    "RET_INT_REG_COUNT", "RET_FLOAT_REG_COUNT", "RET_BUFFER_ALIGN",
    "MAX_ARGS", "MAX_ARRAY_COUNT", "MAX_RECORD_FIELDS", "MAX_TYPE_SIZE",
    "MAX_RET_BUFFER_SIZE",
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
    "encode", "decode", "pack_call", "unpack_call",
    # 返回值
    "ReturnPlan", "ReturnBuffer", "ReturnImage",
    "plan_return", "pack_return", "unpack_return", "simulate_call",
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

RET_INT_REG_COUNT = 2    # 整数类返回寄存器个数
RET_FLOAT_REG_COUNT = 2  # 浮点类返回寄存器个数
RET_BUFFER_ALIGN = 16    # 间接返回缓冲区地址对齐

MAX_ARGS = 1024                 # 单次调用最多参数个数
MAX_ARRAY_COUNT = 65536         # 数组最大元素数
MAX_RECORD_FIELDS = 256         # 记录最大字段数
MAX_TYPE_SIZE = 1 << 20         # 任一类型完整布局最大字节数（1 MiB）
MAX_RET_BUFFER_SIZE = MAX_TYPE_SIZE  # 返回缓冲区容量上限（1 MiB）

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
    ret_plan: 返回值安放方案（未指定返回类型时为 None）。
    hidden_ptr: 间接返回时隐藏指针实参的安放结果（否则为 None）；它先于
      所有用户参数占用一个整数参数寄存器，但不计入 placements。
    """

    int_regs: tuple
    float_regs: tuple
    stack: bytes
    placements: tuple[Placement, ...]
    ret_plan: "ReturnPlan | None" = None
    hidden_ptr: Placement | None = None


def pack_call(args, ret_type=None, ret_buffer=None) -> CallImage:
    """把 ``[(Type, value), ...]`` 编组为寄存器 + 栈映像。

    逐参数：size 为 0 的聚合不占资源；超过 16 字节走栈；否则按块申请
    寄存器，寄存器不足时整体走 16 字节对齐的栈且不消耗寄存器。
    任一参数非法时整体抛错，不返回部分结果。

    指定 ``ret_type`` 时同时规划返回值：超过 16 字节的返回聚合要求
    ``ret_buffer``（ReturnBuffer），其大小、对齐、范围在编码任何参数
    之前先验证；隐藏指针（缓冲区地址）先占整数参数寄存器 0，再安排
    原参数。返回类型不需要缓冲区时传入 ``ret_buffer`` 同样报错。
    """
    if not isinstance(args, (list, tuple)):
        raise InvalidValueError(
            f"args must be a list/tuple of (Type, value) pairs, got {args!r}")
    if len(args) > MAX_ARGS:
        raise InvalidValueError(
            f"too many arguments: {len(args)} > limit {MAX_ARGS}")

    int_regs: list = [None] * INT_REG_COUNT
    float_regs: list = [None] * FLOAT_REG_COUNT
    stack = bytearray()
    placements: list[Placement] = []
    next_int = 0
    next_float = 0

    # 返回值规划与缓冲区验证：先于任何参数编码，失败不留部分结果。
    ret_plan: ReturnPlan | None = None
    hidden_ptr: Placement | None = None
    if ret_type is not None:
        _check_type(ret_type)
        ret_plan = plan_return(ret_type)
        if ret_plan.kind == "indirect":
            buf = _validate_return_buffer(ret_plan, ret_buffer)
            int_regs[0] = buf.address
            next_int = 1
            hidden_ptr = Placement("registers", (INTEGER,), (0,), (), None, 8)
        elif ret_buffer is not None:
            raise InvalidValueError(
                f"return type {ret_type.name} is returned in registers; "
                "a return buffer must not be supplied", "ret_buffer")
    elif ret_buffer is not None:
        raise InvalidValueError(
            "ret_buffer supplied without ret_type", "ret_buffer")

    def to_stack(data: bytes, cls: tuple[str, ...], size: int) -> Placement:
        off = _align_up(len(stack), STACK_ALIGN)
        stack.extend(b"\x00" * (off - len(stack)))
        stack.extend(data)
        return Placement("stack", cls, (), (), off, size)

    for i, item in enumerate(args):
        if not (isinstance(item, tuple) and len(item) == 2):
            raise InvalidValueError(
                f"each argument must be a (Type, value) pair, got {item!r}",
                f"arg[{i}]")
        t, value = item
        try:
            _check_type(t)
            lay = layout(t)
        except ABIError as exc:
            raise type(exc)(exc.message, f"arg[{i}]") from None
        data = encode(t, value, lay, path=f"arg[{i}]")
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

    stack.extend(b"\x00" * (-len(stack) % STACK_ALIGN))
    return CallImage(tuple(int_regs), tuple(float_regs), bytes(stack),
                     tuple(placements), ret_plan, hidden_ptr)


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
# 返回值编组：返回寄存器 / 有界返回缓冲区
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReturnPlan:
    """一个返回类型的安放方案。

    kind: ``"registers"``（标量或 ≤16 字节聚合，走返回寄存器）/
    ``"indirect"``（>16 字节聚合，经调用方提供的缓冲区间接返回）/
    ``"none"``（空记录，不占任何返回寄存器）。
    """

    kind: str
    eightbytes: tuple[str, ...]
    size: int
    align: int


@dataclass(frozen=True)
class ReturnBuffer:
    """调用方为间接返回提供的有界缓冲区。

    address: 缓冲区起始地址（须按 RET_BUFFER_ALIGN 对齐的 64 位无符号值）；
    size: 缓冲区容量字节数（须 ≥ 返回类型 size，且 ≤ MAX_RET_BUFFER_SIZE）。
    """

    address: int
    size: int


@dataclass(frozen=True)
class ReturnImage:
    """一次返回的完整编组映像。

    int_regs / float_regs: 返回寄存器映像（未用槽位为 None）。
    memory: 间接返回时被调方写入缓冲区地址处的字节串（其余情形为 b""）。
    """

    plan: ReturnPlan
    int_regs: tuple
    float_regs: tuple
    memory: bytes


def plan_return(ret_type: Type) -> ReturnPlan:
    """规划返回类型的安放方案（与参数同一套块分类）。"""
    _check_type(ret_type)
    lay = layout(ret_type)
    cls = classify(ret_type)
    if not cls:
        return ReturnPlan("none", (), 0, 1)
    if cls == (MEMORY,):
        return ReturnPlan("indirect", cls, lay.size, lay.align)
    return ReturnPlan("registers", cls, lay.size, lay.align)


def _validate_return_buffer(plan: ReturnPlan, buf) -> ReturnBuffer:
    """验证间接返回缓冲区的大小、对齐与范围。失败抛错，调用方不得写部分结果。"""
    path = "ret_buffer"
    if not isinstance(buf, ReturnBuffer):
        raise InvalidValueError(
            f"return type needs an indirect buffer: expected a ReturnBuffer, "
            f"got {buf!r}", path)
    if isinstance(buf.address, bool) or not isinstance(buf.address, int):
        raise InvalidValueError(
            f"buffer address must be an int (bool is not accepted), "
            f"got {buf.address!r}", path)
    if not 0 <= buf.address <= (1 << 64) - 1:
        raise InvalidValueError(
            f"buffer address {buf.address} out of range [0, 2**64 - 1]", path)
    if buf.address % RET_BUFFER_ALIGN != 0:
        raise InvalidValueError(
            f"buffer address {buf.address:#x} is not aligned to "
            f"{RET_BUFFER_ALIGN}", path)
    if isinstance(buf.size, bool) or not isinstance(buf.size, int):
        raise InvalidValueError(
            f"buffer size must be an int (bool is not accepted), "
            f"got {buf.size!r}", path)
    if not 0 < buf.size <= MAX_RET_BUFFER_SIZE:
        raise InvalidValueError(
            f"buffer size {buf.size} out of range (0, {MAX_RET_BUFFER_SIZE}]",
            path)
    if buf.size < plan.size:
        raise InvalidValueError(
            f"buffer size {buf.size} < return type size {plan.size}", path)
    if buf.address + plan.size > 1 << 64:
        raise InvalidValueError(
            f"buffer range [{buf.address:#x}, {buf.address + plan.size:#x}) "
            "overflows the 64-bit address space", path)
    return buf


def pack_return(ret_type: Type, value, plan: ReturnPlan | None = None
                ) -> ReturnImage:
    """把返回值编组为返回寄存器映像或待写入缓冲区的字节串。

    值先经 ``encode`` 完整校验与编码，非法值抛错且不产生任何部分结果；
    padding 一律写零。
    """
    _check_type(ret_type)
    if plan is None:
        plan = plan_return(ret_type)
    elif plan != plan_return(ret_type):
        raise InvalidValueError(
            f"plan {plan!r} does not match return type {ret_type.name}")
    data = encode(ret_type, value, path="return")

    if plan.kind == "none":
        return ReturnImage(plan, (None,) * RET_INT_REG_COUNT,
                           (None,) * RET_FLOAT_REG_COUNT, b"")
    if plan.kind == "indirect":
        return ReturnImage(plan, (None,) * RET_INT_REG_COUNT,
                           (None,) * RET_FLOAT_REG_COUNT, data)

    int_regs: list = [None] * RET_INT_REG_COUNT
    float_regs: list = [None] * RET_FLOAT_REG_COUNT
    ni = nf = 0
    for k, c in enumerate(plan.eightbytes):
        chunk = data[8 * k:8 * k + 8].ljust(8, b"\x00")
        if c == INTEGER:
            int_regs[ni] = int.from_bytes(chunk, "little")
            ni += 1
        else:
            float_regs[nf] = struct.unpack("<d", chunk)[0]
            nf += 1
    return ReturnImage(plan, tuple(int_regs), tuple(float_regs), b"")


def unpack_return(ret_type: Type, image: ReturnImage,
                  plan: ReturnPlan | None = None):
    """调用方按返回类型从 ``image`` 读取返回值（``pack_return`` 的逆）。

    只按字段布局读取有效字节，padding 不作为值解释。
    """
    _check_type(ret_type)
    if not isinstance(image, ReturnImage):
        raise InvalidValueError(f"expected a ReturnImage, got {image!r}")
    if plan is None:
        plan = plan_return(ret_type)
    if image.plan != plan:
        raise InvalidValueError(
            f"image plan {image.plan!r} does not match return type "
            f"{ret_type.name} plan {plan!r}")
    lay = layout(ret_type)

    if plan.kind == "none":
        return decode(ret_type, b"", lay, "return")
    if plan.kind == "indirect":
        if len(image.memory) != plan.size:
            raise InvalidValueError(
                f"return buffer content is {len(image.memory)} bytes, "
                f"expected {plan.size}", "return")
        return decode(ret_type, image.memory, lay, "return")

    chunks: list[bytes] = []
    ni = nf = 0
    for c in plan.eightbytes:
        if c == INTEGER:
            v = image.int_regs[ni] if ni < len(image.int_regs) else None
            ni += 1
            if v is None:
                raise InvalidValueError(
                    "return integer register slot is empty", "return")
            chunks.append(v.to_bytes(8, "little"))
        else:
            v = image.float_regs[nf] if nf < len(image.float_regs) else None
            nf += 1
            if v is None:
                raise InvalidValueError(
                    "return float register slot is empty", "return")
            chunks.append(struct.pack("<d", v))
    return decode(ret_type, b"".join(chunks)[:lay.size], lay, "return")


def simulate_call(func, args, ret_type: Type, ret_buffer=None):
    """完整调用流程：参数编组 → 被调方解参与计算 → 返回编组 → 调用方读取。

    ``func`` 接收解组后的实参并返回结果；返回值按 ``ret_type`` 编组，
    间接返回时写入 ``ret_buffer`` 指向的缓冲区（以 ReturnImage.memory
    表示写入的字节）。任一步失败整体抛错，不留下部分结果。
    """
    if not callable(func):
        raise InvalidValueError(f"func must be callable, got {func!r}")
    image = pack_call(args, ret_type=ret_type, ret_buffer=ret_buffer)
    values = unpack_call([t for t, _ in args], image)
    result = func(*values)
    ret_image = pack_return(ret_type, result, image.ret_plan)
    return unpack_return(ret_type, ret_image, image.ret_plan)
