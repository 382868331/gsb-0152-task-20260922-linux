"""abi_marshal — 聚合参数的合成 ABI 编组（离线外部函数接口工程）。

本模块包含两代明示 ABI 契约（均不宣称兼容任何真实平台 ABI，也不读取
主机 C 结构布局——所有布局均由显式描述符推导）：

* **v1（LEGACY，默认）**：原始契约，字节行为冻结——
  小端字节序；指针 8 字节（无符号）；整数 1/2/4/8 字节（有符号或无符号）；
  浮点 8 字节（IEEE-754 binary64，入口拒绝 NaN/Infinity）；记录按自然对齐
  布局，对齐上限 8；空记录 size=0、align=1；数组显式长度连续排列；
  聚合按 8 字节块分类：块内含任意整数/指针字节则 INTEGER，否则 FLOAT；
  超过 16 字节整体走栈（MEMORY）；整数寄存器 4 个、浮点寄存器 2 个；
  聚合所需寄存器不足时整体走 16 字节对齐的栈且不消耗任何寄存器；
  空记录不消耗寄存器也不消耗栈。

* **v2（显式描述符 ``AbiDescriptor``）**：在同一数据模型上扩展——
  显式字节序（描述符 A 小端 / 描述符 B 大端，两种相反布局）、显式标量
  对齐与聚合对齐上限、显式寄存器槽规则（数量、128 位整数占槽与奇偶
  配对）；128 位整数、32 位浮点、联合体（活动成员标签）、可跨存储单元
  的位域、可变参数默认提升；返回值间接区（sret）、可变参数寄存器保存区
  与按位共享存储单元（cell）。规划器 plan_call 与编解码器
  encode_call/decode_call 分离，输出逐位布局证据（BitEvidence 证书）；
  bridge_call/bridge_into 在两种布局间映像转换：先解码为规范值再按目标
  描述符编组，保留有符号边界位与 NaN 原始载荷位，失败不部分改写目标。
  v2 完整契约见 README「v2 契约」一节。

旧入口（layout/classify/encode/decode/pack_call/unpack_call）默认仍为
v1；传入描述符（关键字 abi 或 plan_call 系列）即走 v2。旧调用、旧异常
类型与旧确定性结果保持不变。

错误语义：所有非法输入抛 ABIError 子类，携带可定位 path
（如 arg[2].samples[3]、vararg[1]、cell 'buf'）。
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = [
    # 常量（v1）
    "INTEGER", "FLOAT", "MEMORY",
    "INT_REG_COUNT", "FLOAT_REG_COUNT", "STACK_ALIGN", "MAX_EIGHTBYTE_SIZE",
    "MAX_ARGS", "MAX_ARRAY_COUNT", "MAX_RECORD_FIELDS", "MAX_TYPE_SIZE",
    # v2
    "DEFAULT_BUDGET", "FRAME_BASE",
    # 错误
    "ABIError", "InvalidTypeError", "RecursiveTypeError",
    "InvalidValueError", "IntegerOverflowError", "NonFiniteFloatError",
    "BitFieldError", "TagError", "PromotionError", "BridgeError",
    "BudgetExhausted",
    # 类型（v1）
    "Type", "IntType", "FloatType", "PtrType", "ArrayType", "RecordType",
    "INT8", "INT16", "INT32", "INT64",
    "UINT8", "UINT16", "UINT32", "UINT64",
    "DOUBLE", "PTR",
    # 类型（v2 新增）
    "IntNType", "Float32Type", "BitField", "UnionType",
    "INT128", "UINT128", "FLOAT32", "Tagged", "CellPtr",
    # v2 描述符
    "AbiDescriptor", "DESCRIPTOR_A", "DESCRIPTOR_B", "V2_LE", "V2_BE",
    # 布局 / 分类
    "Layout", "FieldLayout", "MemberLayout", "BitSegment",
    "layout", "classify",
    # v1 编组
    "Placement", "CallImage",
    "encode", "decode", "pack_call", "unpack_call",
    # v2 规划 / 编解码 / 证书 / 桥接
    "Signature", "V2Placement", "ReturnInfo", "SaveArea", "CellInfo",
    "Plan", "V2CallImage", "CallValues",
    "BitEvidence", "Certificate", "check_certificate",
    "plan_call", "encode_call", "decode_call",
    "bridge_call", "bridge_into",
]

# ===========================================================================
# ABI 常量与输入上限（v1）
# ===========================================================================

INTEGER = "INTEGER"
FLOAT = "FLOAT"
MEMORY = "MEMORY"

INT_REG_COUNT = 4
FLOAT_REG_COUNT = 2
STACK_ALIGN = 16
MAX_EIGHTBYTE_SIZE = 16  # v1：聚合超过 16 字节整体走栈

MAX_ARGS = 1024                 # v1 单次调用最多参数个数
MAX_ARRAY_COUNT = 65536
MAX_RECORD_FIELDS = 256
MAX_TYPE_SIZE = 1 << 20         # 1 MiB

# v2
DEFAULT_BUDGET = 20_000_000     # plan_call 工作单位预算（位）
FRAME_BASE = 0x0000_7F00_0000_0000  # 合成帧相对地址空间的 CellPtr 基址

_INT_SIZES = (1, 2, 4, 8)
_ENDIAN_CHAR = {"little": "<", "big": ">"}


# ===========================================================================
# 错误层级
# ===========================================================================

class ABIError(Exception):
    """本模块所有错误的基类。``path`` 给出可定位的出错位置。"""

    def __init__(self, message: str, path: str | None = None):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}" if path else message)


class InvalidTypeError(ABIError):
    """类型定义本身非法（未定义记录、字段重复、尺寸超限、描述符不支持等）。"""


class RecursiveTypeError(InvalidTypeError):
    """记录/联合类型直接或间接包含自身，布局无限大。"""


class InvalidValueError(ABIError):
    """值与类型不匹配（类型错误、长度不符、字段缺失/多余等）。"""


class IntegerOverflowError(InvalidValueError):
    """整数/指针/位域值超出目标类型可表示范围。"""


class NonFiniteFloatError(InvalidValueError):
    """v1 入口：浮点为 NaN 或 Infinity（v2 允许并保留其原始位模式）。"""


class BitFieldError(InvalidTypeError):
    """位域重叠、越界，或描述符禁止跨存储单元却发生跨越。"""


class TagError(InvalidValueError):
    """联合体活动成员标签不存在，或标签值超出标签整数类型范围。"""


class PromotionError(InvalidValueError):
    """可变参数默认提升无法在目标布局表示（如 128 位整数经可变参数传递）。"""


class BudgetExhausted(ABIError):
    """规划器显式预算耗尽（与“已证明无解”的拒绝严格区分）。"""


class BridgeError(ABIError):
    """两种布局间桥接被确定性拒绝。

    reason: "value" / "promotion" / "tag" / "unidentifiable" / "budget"。
    """

    def __init__(self, message: str, reason: str, path: str | None = None):
        self.reason = reason
        super().__init__(message, path)


# ===========================================================================
# 类型描述
# ===========================================================================

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
                f"integer size must be one of {_INT_SIZES}, got {self.size!r}; "
                "128-bit integers use IntNType/INT128/UINT128 (v2)")
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
class IntNType(Type):
    """宽整数（v2）：当前仅支持 128 位（16 字节）。v1 入口显式拒绝。"""

    size: int
    signed: bool

    def __post_init__(self):
        if self.size != 16:
            raise InvalidTypeError(
                f"wide integer size must be 16 (128-bit), got {self.size!r}")
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
class Float32Type(Type):
    """4 字节 IEEE-754 binary32 浮点（v2；作为可变参数时默认提升为 f64）。"""

    @property
    def name(self) -> str:
        return "f32"


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


@dataclass(frozen=True)
class BitField(Type):
    """位域叶子（v2，只能作为记录成员）。

    width: 位宽 1..64；signed: 二补码有符号。连续的位域成员组成一个位域段
    （run），按 8 位存储单元（字节）线性分配；描述符允许时单个位域可跨
    存储单元边界。记录字段可给第三项 pin（段内线性位偏移）做显式定位，
    用于制造并检测“重叠位域”。
    """

    width: int
    signed: bool = False

    def __post_init__(self):
        if isinstance(self.width, bool) or not isinstance(self.width, int):
            raise InvalidTypeError(
                f"bitfield width must be an int, got {self.width!r}")
        if not 1 <= self.width <= 64:
            raise InvalidTypeError(
                f"bitfield width {self.width} out of range [1, 64]")
        if not isinstance(self.signed, bool):
            raise InvalidTypeError(f"signed must be bool, got {self.signed!r}")

    @property
    def name(self) -> str:
        return f"{'i' if self.signed else 'u'}bf{self.width}"


class RecordType(Type):
    """命名字段记录。支持先建壳后 define()，以便表达（并拒绝）循环。

    define 字段项为 (name, Type)；Type 为 BitField 时可给第三项 pin。
    """

    __slots__ = ("_name", "_fields")

    def __init__(self, name: str):
        if not isinstance(name, str) or not name:
            raise InvalidTypeError(f"record name must be a non-empty str, got {name!r}")
        self._name = name
        self._fields: tuple[tuple[str, Type, int | None], ...] | None = None

    @property
    def name(self) -> str:
        return self._name

    def define(self, fields) -> None:
        """一次性定义字段序列。只能定义一次。"""
        if self._fields is not None:
            raise InvalidTypeError(f"record {self._name!r} is already defined")
        items = tuple(fields)
        if len(items) > MAX_RECORD_FIELDS:
            raise InvalidTypeError(
                f"record {self._name!r} has {len(items)} fields, "
                f"limit is {MAX_RECORD_FIELDS}")
        seen: set[str] = set()
        norm: list[tuple[str, Type, int | None]] = []
        for item in items:
            if not isinstance(item, tuple) or len(item) not in (2, 3):
                raise InvalidTypeError(
                    f"record {self._name!r}: field must be a (name, Type) or "
                    f"(name, BitField, pin) pair, got {item!r}")
            fname, ftype = item[0], item[1]
            pin = item[2] if len(item) == 3 else None
            if not isinstance(fname, str) or not fname:
                raise InvalidTypeError(
                    f"record {self._name!r}: field name must be a non-empty str, "
                    f"got {fname!r}")
            if fname in seen:
                raise InvalidTypeError(
                    f"record {self._name!r}: duplicate field name {fname!r}")
            seen.add(fname)
            _check_type(ftype)
            if pin is not None:
                if not isinstance(ftype, BitField):
                    raise InvalidTypeError(
                        f"record {self._name!r} field {fname!r}: pin is only "
                        "valid on a BitField member")
                if isinstance(pin, bool) or not isinstance(pin, int) or pin < 0:
                    raise InvalidTypeError(
                        f"record {self._name!r} field {fname!r}: pin must be a "
                        f"non-negative int, got {pin!r}")
            norm.append((fname, ftype, pin))
        self._fields = tuple(norm)

    @property
    def fields(self) -> tuple[tuple[str, Type, int | None], ...]:
        if self._fields is None:
            raise InvalidTypeError(f"record {self._name!r} is not defined yet")
        return self._fields


class UnionType(Type):
    """联合体（v2）：各成员共享同一存储，外加一个整数标签。

    字节布局：[tag: tag_type][storage: 最大成员尺寸]，storage 按联合体
    对齐起始，整体尺寸向上取齐到联合体对齐。活动成员由标签值选择；
    标签值在 define 时逐成员显式声明。值用 Tagged(成员名, 成员值) 表示。
    """

    __slots__ = ("_name", "_members", "_tag_type")

    def __init__(self, name: str):
        if not isinstance(name, str) or not name:
            raise InvalidTypeError(f"union name must be a non-empty str, got {name!r}")
        self._name = name
        self._members: tuple[tuple[str, Type, int], ...] | None = None
        self._tag_type: IntType | None = None

    @property
    def name(self) -> str:
        return self._name

    def define(self, members, tag_type: IntType | None = None) -> None:
        if self._members is not None:
            raise InvalidTypeError(f"union {self._name!r} is already defined")
        items = tuple(members)
        if not items:
            raise InvalidTypeError(f"union {self._name!r} must have at least one member")
        if len(items) > MAX_RECORD_FIELDS:
            raise InvalidTypeError(
                f"union {self._name!r} has {len(items)} members, "
                f"limit is {MAX_RECORD_FIELDS}")
        tag_type = tag_type or UINT8
        if not isinstance(tag_type, IntType):
            raise InvalidTypeError(
                f"union {self._name!r}: tag_type must be a 1/2/4/8-byte IntType, "
                f"got {tag_type!r}")
        lo, hi = tag_type.range
        seen_names: set[str] = set()
        seen_tags: set[int] = set()
        norm: list[tuple[str, Type, int]] = []
        for item in items:
            if not (isinstance(item, tuple) and len(item) == 3):
                raise InvalidTypeError(
                    f"union {self._name!r}: member must be a (name, Type, tag) "
                    f"triple, got {item!r}")
            mname, mtype, tag = item
            if not isinstance(mname, str) or not mname:
                raise InvalidTypeError(
                    f"union {self._name!r}: member name must be a non-empty str, "
                    f"got {mname!r}")
            if mname in seen_names:
                raise InvalidTypeError(
                    f"union {self._name!r}: duplicate member name {mname!r}")
            seen_names.add(mname)
            _check_type(mtype)
            if isinstance(mtype, BitField):
                raise InvalidTypeError(
                    f"union {self._name!r} member {mname!r}: a bare BitField "
                    "cannot be a union member (wrap it in a record)")
            if isinstance(tag, bool) or not isinstance(tag, int):
                raise InvalidTypeError(
                    f"union {self._name!r} member {mname!r}: tag must be an int, "
                    f"got {tag!r}")
            if not lo <= tag <= hi:
                raise InvalidTypeError(
                    f"union {self._name!r} member {mname!r}: tag {tag} out of "
                    f"range [{lo}, {hi}] for {tag_type.name}")
            if tag in seen_tags:
                raise InvalidTypeError(
                    f"union {self._name!r}: duplicate tag value {tag}")
            seen_tags.add(tag)
            norm.append((mname, mtype, tag))
        self._members = tuple(norm)
        self._tag_type = tag_type

    @property
    def members(self) -> tuple[tuple[str, Type, int], ...]:
        if self._members is None:
            raise InvalidTypeError(f"union {self._name!r} is not defined yet")
        return self._members

    @property
    def tag_type(self) -> IntType:
        if self._tag_type is None:
            raise InvalidTypeError(f"union {self._name!r} is not defined yet")
        return self._tag_type


@dataclass(frozen=True)
class Tagged:
    """联合体值：Tagged(成员名, 成员值)。"""

    member: str
    value: object


@dataclass(frozen=True)
class CellPtr:
    """指针实参值：指向命名共享存储单元；编解码时按帧地址回填/重基址。"""

    cell: str


# ---------------------------------------------------------------------------
# 标量常量
# ---------------------------------------------------------------------------

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

INT128 = IntNType(16, True)
UINT128 = IntNType(16, False)
FLOAT32 = Float32Type()


def _check_type(t) -> None:
    if not isinstance(t, Type):
        raise InvalidTypeError(f"expected an ABI Type, got {t!r}")


def _align_up(offset: int, align: int) -> int:
    return (offset + align - 1) // align * align


# ===========================================================================
# v2 描述符
# ===========================================================================

@dataclass(frozen=True)
class AbiDescriptor:
    """目标布局的显式声明（v2）。新布局只能由此推导，不查主机 struct。

    name: 描述符名。
    endian: "little" / "big"。
    int_reg_count / float_reg_count: 寄存器槽数量。
    stack_align: 栈参数 / 帧内各区域最小对齐（2 的幂）。
    max_reg_bytes: 聚合经寄存器传递的尺寸上限；超过即 MEMORY 整体走栈。
    max_agg_align: 聚合对齐上限（记录/联合/数组对齐超过则截断到此值）。
    align_int: 整数尺寸（1/2/4/8/16 字节）→ 标量对齐。
    align_float: 浮点尺寸（4/8）→ 标量对齐。
    align_ptr: 指针对齐（固定 8）。
    bitfield_unit_bits: 位域存储单元位宽（仅允许 8）。
    bitfield_span: 是否允许位域跨存储单元。
    i128_even_pair: 128 位整数占两连续整数槽时是否要求偶数号起配对
                    （奇数号槽留空，且不回溯给其他实参）。
    sret_threshold: 聚合返回值尺寸超过此值（或被分类为 MEMORY）则经隐式
                    sret 间接区返回。
    max_args / max_varargs: 定位参数 / 可变参数个数上限。
    """

    name: str
    endian: str
    int_reg_count: int = 6
    float_reg_count: int = 4
    stack_align: int = 16
    max_reg_bytes: int = 32
    max_agg_align: int = 16
    align_int: dict = field(default_factory=lambda: {1: 1, 2: 2, 4: 4, 8: 8, 16: 8})
    align_float: dict = field(default_factory=lambda: {4: 4, 8: 8})
    align_ptr: int = 8
    bitfield_unit_bits: int = 8
    bitfield_span: bool = True
    i128_even_pair: bool = False
    sret_threshold: int = 16
    max_args: int = 32
    max_varargs: int = 32

    def __post_init__(self):
        if self.endian not in _ENDIAN_CHAR:
            raise InvalidTypeError(
                f"endian must be 'little' or 'big', got {self.endian!r}")
        if self.stack_align not in (1, 2, 4, 8, 16, 32, 64):
            raise InvalidTypeError(f"bad stack_align {self.stack_align}")
        for key in ("int_reg_count", "float_reg_count", "max_reg_bytes",
                    "max_agg_align", "sret_threshold", "max_args",
                    "max_varargs"):
            v = getattr(self, key)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise InvalidTypeError(f"{key} must be a positive int, got {v!r}")
        if self.max_reg_bytes % 8:
            raise InvalidTypeError("max_reg_bytes must be a multiple of 8")
        if self.bitfield_unit_bits != 8:
            raise InvalidTypeError("bitfield_unit_bits must be 8")
        for sz, al in self.align_int.items():
            if sz not in (1, 2, 4, 8, 16) or not (1 <= al <= sz) \
                    or (sz // al) * al != sz:
                raise InvalidTypeError(f"bad integer alignment {sz}:{al}")
        for sz, al in self.align_float.items():
            if sz not in (4, 8) or not (1 <= al <= sz) \
                    or (sz // al) * al != sz:
                raise InvalidTypeError(f"bad float alignment {sz}:{al}")
        if self.align_ptr != 8:
            raise InvalidTypeError("align_ptr must be 8")
        object.__setattr__(self, "_lcache", {})

    @property
    def endian_char(self) -> str:
        return _ENDIAN_CHAR[self.endian]

    def scalar_align(self, t: Type) -> int:
        if isinstance(t, (IntType, IntNType)):
            return self.align_int[t.size]
        if isinstance(t, Float32Type):
            return self.align_float[4]
        if isinstance(t, FloatType):
            return self.align_float[8]
        if isinstance(t, PtrType):
            return self.align_ptr
        raise InvalidTypeError(f"{t!r} has no scalar alignment")  # pragma: no cover


# 描述符 A：小端；描述符 B：大端（两种相反字节序），槽规则也不同。
DESCRIPTOR_A = AbiDescriptor(
    name="A-little",
    endian="little",
    int_reg_count=6,
    float_reg_count=4,
    stack_align=16,
    max_reg_bytes=32,
    max_agg_align=16,
    align_int={1: 1, 2: 2, 4: 4, 8: 8, 16: 8},
    align_float={4: 4, 8: 8},
    i128_even_pair=False,
    sret_threshold=16,
)
DESCRIPTOR_B = AbiDescriptor(
    name="B-big",
    endian="big",
    int_reg_count=4,
    float_reg_count=2,
    stack_align=16,
    max_reg_bytes=16,
    max_agg_align=16,
    align_int={1: 1, 2: 2, 4: 4, 8: 8, 16: 16},
    align_float={4: 4, 8: 8},
    i128_even_pair=True,
    sret_threshold=8,
)
V2_LE = DESCRIPTOR_A
V2_BE = DESCRIPTOR_B


# ===========================================================================
# 布局结果
# ===========================================================================

@dataclass(frozen=True)
class BitSegment:
    """位域的一段：位于所属记录相对字节 byte 内；物理位 (lsb+k),
    0<=k<count，承载值的第 (value_bit+k) 位（rev=False）或
    第 (value_bit+count-1-k) 位（rev=True，大端字节内反向）。"""

    byte: int
    lsb: int
    count: int
    value_bit: int
    rev: bool = False


@dataclass(frozen=True)
class FieldLayout:
    """记录中一个字段的布局（v2 位域字段后三项非空）。"""

    name: str
    offset: int
    type: Type
    layout: "Layout"
    bit_width: int | None = None
    bit_signed: bool = False
    bit_segments: tuple[BitSegment, ...] = ()


@dataclass(frozen=True)
class MemberLayout:
    """联合体一个成员的布局。"""

    name: str
    tag: int
    type: Type
    layout: "Layout"


@dataclass(frozen=True)
class Layout:
    """一个类型在某一契约下的完整字节布局。"""

    type: Type
    size: int
    align: int
    fields: tuple[FieldLayout, ...] = ()
    members: tuple[MemberLayout, ...] = ()
    tag_layout: "Layout | None" = None
    tag_offset: int = 0
    storage_offset: int = 0


# ===========================================================================
# v1 布局
# ===========================================================================

def layout(t: Type, *, abi: AbiDescriptor | None = None,
           _visiting: frozenset[int] | None = None) -> Layout:
    """计算类型 t 的字节布局。

    abi=None 时为 v1（小端、对齐上限 8）；传入 v2 描述符走 v2 布局。
    循环类型抛 RecursiveTypeError。
    """
    _check_type(t)
    if abi is not None:
        return _layout_v2(abi, t)
    if _visiting is None:
        _visiting = frozenset()

    if isinstance(t, IntType):
        return Layout(t, t.size, t.size)
    if isinstance(t, FloatType):
        return Layout(t, 8, 8)
    if isinstance(t, PtrType):
        return Layout(t, 8, 8)
    if isinstance(t, (IntNType, Float32Type, BitField, UnionType)):
        raise InvalidTypeError(
            f"{t.name} requires a v2 descriptor (layout(t, abi=DESCRIPTOR_A/B))")

    if isinstance(t, ArrayType):
        elem_layout = layout(t.elem, _visiting=_visiting)
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
        fields = t.fields
        visiting = _visiting | {id(t)}
        offset = 0
        align = 1
        field_layouts: list[FieldLayout] = []
        for fname, ftype, _pin in fields:
            flay = layout(ftype, _visiting=visiting)
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


# ===========================================================================
# v2 布局
# ===========================================================================

def _layout_v2(desc: AbiDescriptor, t: Type,
               visiting: frozenset[int] | None = None) -> Layout:
    cache = desc._lcache  # type: ignore[attr-defined]
    if visiting is None:
        hit = cache.get(id(t))
        if hit is not None:
            return hit
        visiting = frozenset()
    else:
        hit = cache.get((id(t), visiting))
        if hit is not None:
            return hit

    if isinstance(t, (IntType, IntNType)):
        res = Layout(t, t.size, desc.scalar_align(t))
    elif isinstance(t, (FloatType, Float32Type)):
        size = 8 if isinstance(t, FloatType) else 4
        res = Layout(t, size, desc.scalar_align(t))
    elif isinstance(t, PtrType):
        res = Layout(t, 8, desc.align_ptr)
    elif isinstance(t, BitField):
        raise InvalidTypeError(
            "a bare BitField has no byte layout (it must be a record member)")
    elif isinstance(t, ArrayType):
        if id(t) in visiting:
            raise RecursiveTypeError(
                f"recursive array element type {t.elem.name!r}")
        elem_layout = _layout_v2(desc, t.elem, visiting)
        size = elem_layout.size * t.count
        if size > MAX_TYPE_SIZE:
            raise InvalidTypeError(
                f"type size {size} exceeds limit {MAX_TYPE_SIZE}")
        align = min(elem_layout.align if t.count > 0 else 1,
                    desc.max_agg_align)
        res = Layout(t, size, align)
    elif isinstance(t, RecordType):
        if id(t) in visiting:
            raise RecursiveTypeError(
                f"recursive record type {t.name!r} has infinite layout")
        fields = t.fields
        nxt = visiting | {id(t)}
        offset = 0
        align = 1
        flayouts: list[FieldLayout] = []
        i = 0
        while i < len(fields):
            if not isinstance(fields[i][1], BitField):
                fname, ftype, _pin = fields[i]
                flay = _layout_v2(desc, ftype, nxt)
                al = min(flay.align, desc.max_agg_align)
                align = max(align, al)
                offset = _align_up(offset, al)
                flayouts.append(FieldLayout(fname, offset, ftype, flay))
                offset += flay.size
                i += 1
                continue
            # 收集连续位域段
            run = []
            while i < len(fields) and isinstance(fields[i][1], BitField):
                run.append(fields[i])
                i += 1
            # 第一遍：线性位区间（与字节序无关），检测重叠/越界
            cursor = 0
            intervals: list[tuple[str, BitField, int]] = []
            for fname, bf, pin in run:
                g = cursor if pin is None else pin
                for _n, _b, g0 in intervals:
                    if g < g0 + _b.width and g0 < g + bf.width:
                        raise BitFieldError(
                            f"overlapping bitfield {fname!r}: bits "
                            f"[{g},{g + bf.width}) overlap bits "
                            f"[{g0},{g0 + _b.width}) of {_n!r}",
                            f"record {t.name!r} field {fname!r}")
                if not desc.bitfield_span \
                        and g // 8 != (g + bf.width - 1) // 8:
                    raise BitFieldError(
                        f"bitfield {fname!r} of {bf.width} bits crosses a "
                        "storage-unit boundary but the descriptor forbids it",
                        f"record {t.name!r} field {fname!r}")
                intervals.append((fname, bf, g))
                cursor = max(cursor, g + bf.width)
            run_bytes = (cursor + 7) // 8
            # 第二遍：物理位段映射
            run_start = offset
            for fname, bf, g in intervals:
                segs = _map_bitfield(g, bf.width, run_bytes, desc.endian)
                flayouts.append(FieldLayout(
                    fname, run_start, bf, Layout(bf, run_bytes, 1),
                    bit_width=bf.width, bit_signed=bf.signed,
                    bit_segments=tuple(segs)))
            offset += run_bytes
            align = max(align, 1)
        size = _align_up(offset, align)
        if size > MAX_TYPE_SIZE:
            raise InvalidTypeError(
                f"type size {size} exceeds limit {MAX_TYPE_SIZE}")
        res = Layout(t, size, align, tuple(flayouts))
    elif isinstance(t, UnionType):
        if id(t) in visiting:
            raise RecursiveTypeError(
                f"recursive union type {t.name!r} has infinite layout")
        nxt = visiting | {id(t)}
        tag_lay = Layout(t.tag_type, t.tag_type.size,
                         desc.scalar_align(t.tag_type))
        mlayouts: list[MemberLayout] = []
        storage_size = 0
        storage_align = 1
        for mname, mtype, tag in t.members:
            ml = _layout_v2(desc, mtype, nxt)
            mlayouts.append(MemberLayout(mname, tag, mtype, ml))
            storage_size = max(storage_size, ml.size)
            storage_align = max(storage_align, ml.align)
        storage_align = min(storage_align, desc.max_agg_align)
        storage_off = _align_up(tag_lay.size, storage_align)
        size = _align_up(storage_off + storage_size, storage_align)
        res = Layout(t, size, storage_align, (), tuple(mlayouts),
                     tag_lay, 0, storage_off)
    else:  # pragma: no cover
        raise InvalidTypeError(f"unsupported type {t!r}")

    if visiting:
        cache[(id(t), visiting)] = res
    else:
        cache[id(t)] = res
    return res


def _map_bitfield(g: int, width: int, run_bytes: int,
                  endian: str) -> list[BitSegment]:
    """把线性位区间 [g, g+width) 映射为物理字节段。

    线性位地址按“值的最低位方向”递增；物理映射（合成 ABI 明示规则）：
    little: g -> 字节 g//8 内的位 g%8；
    big:    g -> 字节 (run_bytes-1-g//8) 内的位 (7-g%8)。
    """
    raw: dict[int, list[tuple[int, int]]] = {}
    for j in range(width):
        lin = g + j
        if endian == "little":
            byte, bit = lin // 8, lin % 8
        else:
            byte, bit = run_bytes - 1 - lin // 8, 7 - lin % 8
        raw.setdefault(byte, []).append((bit, j))
    segs: list[BitSegment] = []
    for byte in sorted(raw):
        bits = sorted(raw[byte])
        phy = [b for b, _ in bits]
        vbits = [v for _, v in bits]
        lsb = phy[0]
        count = len(phy)
        base = min(vbits)
        rev = any(vbits[k] != base + k for k in range(count))
        segs.append(BitSegment(byte, lsb, count, base, rev))
    return segs


# ===========================================================================
# 聚合分类（v1/v2 通用，按 8 字节块）
# ===========================================================================

def _spans(t: Type, lay: Layout, base: int, out: list,
           desc: AbiDescriptor | None = None) -> None:
    """把类型所有标量叶子展开为 (偏移, 长度, 类) 区间。"""
    if isinstance(t, (IntType, IntNType, PtrType)):
        out.append((base, lay.size, INTEGER))
    elif isinstance(t, (FloatType, Float32Type)):
        out.append((base, lay.size, FLOAT))
    elif isinstance(t, ArrayType):
        if t.count:
            elay = _layout_v2(desc, t.elem) if desc is not None \
                else layout(t.elem)
            step = elay.size
            for i in range(t.count):
                _spans(t.elem, elay, base + i * step, out, desc)
    elif isinstance(t, RecordType):
        for f in lay.fields:
            if f.bit_width is not None:
                lo = base + f.offset + min(s.byte for s in f.bit_segments)
                hi = base + f.offset + max(s.byte for s in f.bit_segments) + 1
                out.append((lo, hi - lo, INTEGER))
            else:
                _spans(f.type, f.layout, base + f.offset, out, desc)
    elif isinstance(t, UnionType):
        # 标签必为整数类；存储区按所有成员字节的静态并集参与分类。
        out.append((base + lay.tag_offset, lay.tag_layout.size, INTEGER))
        for m in lay.members:
            _spans(m.type, m.layout, base + lay.storage_offset, out, desc)
    else:  # pragma: no cover
        raise InvalidTypeError(f"unsupported type {t!r}")


def classify(t: Type, *, abi: AbiDescriptor | None = None) -> tuple[str, ...]:
    """按 8 字节块分类聚合。

    v1 上限 16 字节；v2 上限由描述符 max_reg_bytes 给出。size 为 0 返回
    空元组；超限返回 (MEMORY,)。块内含任意整数/指针/位域字节即 INTEGER。
    """
    _check_type(t)
    if abi is None:
        lay = layout(t)
        limit = MAX_EIGHTBYTE_SIZE
    else:
        lay = _layout_v2(abi, t)
        limit = abi.max_reg_bytes
    if lay.size == 0:
        return ()
    if lay.size > limit:
        return (MEMORY,)
    spans: list[tuple[int, int, str]] = []
    _spans(t, lay, 0, spans, abi if abi is not None else None)
    classes: list[str] = []
    for chunk_start in range(0, lay.size, 8):
        cls = FLOAT
        for off, size, c in spans:
            if c == INTEGER and off < chunk_start + 8 and off + size > chunk_start:
                cls = INTEGER
                break
        classes.append(cls)
    return tuple(classes)


# ===========================================================================
# 值校验小工具
# ===========================================================================

def _check_int_value(value, size: int, signed: bool, path: str, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidValueError(
            f"expected an int (bool is not accepted), got {value!r}", path)
    bits = size * 8
    lo = -(1 << (bits - 1)) if signed else 0
    hi = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    if not lo <= value <= hi:
        raise IntegerOverflowError(
            f"value {value} out of range [{lo}, {hi}] for {name}", path)


# ===========================================================================
# v1 值编码 / 解码
# ===========================================================================

def encode(t: Type, value, lay: Layout | None = None, path: str = "value",
           *, abi: AbiDescriptor | None = None) -> bytes:
    """把 Python 值按类型 t 的布局编码为字节串（长度恰为 size）。

    abi=None 走 v1（小端，拒绝 NaN/Infinity）；v2 走描述符字节序，
    允许 NaN/Infinity 并保留其原始载荷位。
    """
    _check_type(t)
    if abi is not None:
        return _encode_v2(abi, t, value, lay or _layout_v2(abi, t), path)
    if lay is None:
        lay = layout(t)

    if isinstance(t, IntType):
        _check_int_value(value, lay.size, t.signed, path, t.name)
        return value.to_bytes(lay.size, "little", signed=t.signed)

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
            for i, v in enumerate(value))

    if isinstance(t, RecordType):
        if not isinstance(value, Mapping):
            raise InvalidValueError(
                f"expected a mapping of record fields, got {value!r}", path)
        expected = {f.name for f in lay.fields}
        actual = set(value.keys())
        if expected - actual:
            raise InvalidValueError(
                f"record {t.name!r} missing fields {sorted(expected - actual)}",
                path)
        if actual - expected:
            raise InvalidValueError(
                f"record {t.name!r} has unexpected fields {sorted(actual - expected)}",
                path)
        buf = bytearray(lay.size)
        for f in lay.fields:
            chunk = encode(f.type, value[f.name], f.layout, f"{path}.{f.name}")
            buf[f.offset:f.offset + f.layout.size] = chunk
        return bytes(buf)

    raise InvalidTypeError(
        f"type {t.name!r} requires a v2 descriptor")


def decode(t: Type, data: bytes, lay: Layout | None = None,
           path: str = "value", *, abi: AbiDescriptor | None = None):
    """把 encode 的产物按类型 t 解码回 Python 值。"""
    _check_type(t)
    if abi is not None:
        return _decode_v2(abi, t, bytes(data), lay or _layout_v2(abi, t), path)
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
            for i in range(t.count)]
    if isinstance(t, RecordType):
        return {
            f.name: decode(f.type, data[f.offset:f.offset + f.layout.size],
                           f.layout, f"{path}.{f.name}")
            for f in lay.fields}
    raise InvalidTypeError(
        f"type {t.name!r} requires a v2 descriptor")


# ===========================================================================
# v2 值编码 / 解码（字节级）
# ===========================================================================

def _encode_v2(desc: AbiDescriptor, t: Type, value, lay: Layout,
               path: str) -> bytes:
    ec = desc.endian_char
    if isinstance(t, (IntType, IntNType)):
        _check_int_value(value, lay.size, t.signed, path, t.name)
        return value.to_bytes(lay.size, desc.endian, signed=t.signed)
    if isinstance(t, PtrType):
        if isinstance(value, CellPtr):
            raise InvalidValueError(
                "CellPtr is resolved by the call encoder; wrap the containing "
                f"call instead (at {path})", path)
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidValueError(
                f"expected an int address (bool is not accepted), got {value!r}",
                path)
        if not 0 <= value <= (1 << 64) - 1:
            raise IntegerOverflowError(
                f"pointer value {value} out of range [0, 2**64 - 1]", path)
        return value.to_bytes(8, desc.endian)
    if isinstance(t, FloatType):
        if isinstance(value, bool) or not isinstance(value, float):
            raise InvalidValueError(f"expected a float, got {value!r}", path)
        return struct.pack(ec + "d", value)
    if isinstance(t, Float32Type):
        if isinstance(value, bool) or not isinstance(value, float):
            raise InvalidValueError(f"expected a float, got {value!r}", path)
        packed = struct.pack(ec + "f", value)
        back = struct.unpack(ec + "f", packed)[0]
        if struct.pack(ec + "d", back) != struct.pack(ec + "d", value):
            raise InvalidValueError(
                f"f32 value {value!r} is not exactly representable in "
                "binary32; round it explicitly (variadic f32 performs the "
                "documented default-promotion rounding)", path)
        return packed
    if isinstance(t, ArrayType):
        if not isinstance(value, (list, tuple)):
            raise InvalidValueError(
                f"expected a list/tuple of {t.count} elements, got {value!r}",
                path)
        if len(value) != t.count:
            raise InvalidValueError(
                f"array length {len(value)} != declared count {t.count}", path)
        elay = _layout_v2(desc, t.elem)
        return b"".join(
            _encode_v2(desc, t.elem, v, elay, f"{path}[{i}]")
            for i, v in enumerate(value))
    if isinstance(t, RecordType):
        if not isinstance(value, Mapping):
            raise InvalidValueError(
                f"expected a mapping of record fields, got {value!r}", path)
        expected = {f.name for f in lay.fields}
        actual = set(value.keys())
        if expected - actual:
            raise InvalidValueError(
                f"record {t.name!r} missing fields {sorted(expected - actual)}",
                path)
        if actual - expected:
            raise InvalidValueError(
                f"record {t.name!r} has unexpected fields {sorted(actual - expected)}",
                path)
        buf = bytearray(lay.size)
        for f in lay.fields:
            if f.bit_width is not None:
                _write_bitfield(buf, f.offset, f, value[f.name],
                                f"{path}.{f.name}")
            else:
                chunk = _encode_v2(desc, f.type, value[f.name], f.layout,
                                   f"{path}.{f.name}")
                buf[f.offset:f.offset + f.layout.size] = chunk
        return bytes(buf)
    if isinstance(t, UnionType):
        if not isinstance(value, Tagged):
            raise InvalidValueError(
                f"union {t.name!r} value must be Tagged(member, value), "
                f"got {value!r}", path)
        member = None
        for m in lay.members:
            if m.name == value.member:
                member = m
                break
        if member is None:
            raise TagError(
                f"union {t.name!r} has no active member {value.member!r}; "
                f"known members: {[m.name for m in lay.members]}", path)
        buf = bytearray(lay.size)
        tt = lay.tag_layout.type
        _check_int_value(member.tag, tt.size, tt.signed, path, tt.name)
        buf[lay.tag_offset:lay.tag_offset + tt.size] = \
            member.tag.to_bytes(tt.size, desc.endian, signed=tt.signed)
        chunk = _encode_v2(desc, member.type, value.value, member.layout,
                           f"{path}.<{member.name}>")
        buf[lay.storage_offset:lay.storage_offset + member.layout.size] = chunk
        return bytes(buf)
    raise InvalidTypeError(f"unsupported type {t!r}")  # pragma: no cover


def _write_bitfield(buf: bytearray, base: int, f: FieldLayout, value,
                    path: str) -> None:
    w = f.bit_width
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidValueError(
            f"expected an int for bitfield, got {value!r}", path)
    if f.bit_signed:
        lo, hi = -(1 << (w - 1)), (1 << (w - 1)) - 1
    else:
        lo, hi = 0, (1 << w) - 1
    if not lo <= value <= hi:
        raise IntegerOverflowError(
            f"bitfield value {value} out of range [{lo}, {hi}] for width {w}",
            path)
    uvalue = value & ((1 << w) - 1)
    for seg in f.bit_segments:
        for k in range(seg.count):
            vb = seg.value_bit + (seg.count - 1 - k if seg.rev else k)
            if (uvalue >> vb) & 1:
                buf[base + seg.byte] |= 1 << (seg.lsb + k)


def _decode_v2(desc: AbiDescriptor, t: Type, data: bytes, lay: Layout, path: str):
    if not isinstance(data, (bytes, bytearray)):
        raise InvalidValueError(f"expected bytes, got {data!r}", path)
    if len(data) != lay.size:
        raise InvalidValueError(
            f"byte length {len(data)} != layout size {lay.size} for {t.name}",
            path)
    if isinstance(t, (IntType, IntNType)):
        return int.from_bytes(data, desc.endian, signed=t.signed)
    if isinstance(t, PtrType):
        return int.from_bytes(data, desc.endian)
    if isinstance(t, FloatType):
        return struct.unpack(desc.endian_char + "d", data)[0]
    if isinstance(t, Float32Type):
        return struct.unpack(desc.endian_char + "f", data)[0]
    if isinstance(t, ArrayType):
        step = lay.size // t.count if t.count else 0
        elay = _layout_v2(desc, t.elem)
        return [
            _decode_v2(desc, t.elem,
                       data[i * step:(i + 1) * step], elay, f"{path}[{i}]")
            for i in range(t.count)]
    if isinstance(t, RecordType):
        out = {}
        for f in lay.fields:
            if f.bit_width is not None:
                out[f.name] = _read_bitfield(data, f.offset, f)
            else:
                out[f.name] = _decode_v2(
                    desc, f.type,
                    data[f.offset:f.offset + f.layout.size], f.layout,
                    f"{path}.{f.name}")
        return out
    if isinstance(t, UnionType):
        tt = lay.tag_layout.type
        tag_val = int.from_bytes(
            data[lay.tag_offset:lay.tag_offset + tt.size],
            desc.endian, signed=tt.signed)
        member = None
        for m in lay.members:
            if m.tag == tag_val:
                member = m
                break
        if member is None:
            raise TagError(
                f"union {t.name!r}: tag value {tag_val} does not name any "
                f"member (known tags: {sorted(m.tag for m in lay.members)})",
                path)
        val = _decode_v2(
            desc, member.type,
            data[lay.storage_offset:lay.storage_offset + member.layout.size],
            member.layout, f"{path}.<{member.name}>")
        return Tagged(member.name, val)
    raise InvalidTypeError(f"unsupported type {t!r}")  # pragma: no cover


def _read_bitfield(data: bytes, base: int, f: FieldLayout) -> int:
    uvalue = 0
    for seg in f.bit_segments:
        for k in range(seg.count):
            vb = seg.value_bit + (seg.count - 1 - k if seg.rev else k)
            if (data[base + seg.byte] >> (seg.lsb + k)) & 1:
                uvalue |= 1 << vb
    w = f.bit_width
    if f.bit_signed and (uvalue >> (w - 1)) & 1:
        uvalue -= 1 << w
    return uvalue


# ===========================================================================
# v1 调用编组（行为冻结）
# ===========================================================================

@dataclass(frozen=True)
class Placement:
    """一个实参的安放结果（v1）。"""

    kind: str
    eightbytes: tuple[str, ...]
    int_regs: tuple[int, ...] = ()
    float_regs: tuple[int, ...] = ()
    stack_offset: int | None = None
    size: int = 0


@dataclass(frozen=True)
class CallImage:
    """v1 一次调用的完整编组映像。"""

    int_regs: tuple
    float_regs: tuple
    stack: bytes
    placements: tuple[Placement, ...]


def pack_call(args) -> CallImage:
    """v1：把 [(Type, value), ...] 编组为寄存器 + 栈映像。"""
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
            placements.append(to_stack(data, cls, lay.size))

    stack.extend(b"\x00" * (-len(stack) % STACK_ALIGN))
    return CallImage(tuple(int_regs), tuple(float_regs), bytes(stack),
                     tuple(placements))


def unpack_call(types, image: CallImage) -> list:
    """v1：按 types 从 image 重建实参值列表。"""
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


# ===========================================================================
# v2：签名 / 规划
# ===========================================================================

@dataclass(frozen=True)
class Signature:
    """一次 v2 调用的静态类型签名。

    args: 定位参数类型；varargs: 可变参数类型（编码时做默认提升）；
    ret: 返回类型（None 表示无返回值）；
    cells: 命名共享存储单元 [(名字, 类型), ...]；
    ret_cell: 若给出，返回值从该命名单元读出（调用方用 CellPtr 传其地址）；
              否则聚合返回值超阈值时规划器隐式建立 $sret 单元并把其地址
              作为隐藏首参数传入。
    """

    args: tuple[Type, ...] = ()
    varargs: tuple[Type, ...] = ()
    ret: Type | None = None
    cells: tuple[tuple[str, Type], ...] = ()
    ret_cell: str | None = None

    def __init__(self, args=(), varargs=(), ret=None, cells=(),
                 ret_cell: str | None = None):
        norm = []
        for i, t in enumerate(args):
            _check_type(t)
            if isinstance(t, BitField):
                raise InvalidTypeError(
                    "a bare BitField cannot be a call argument", f"arg[{i}]")
            norm.append(t)
        vnorm = []
        for i, t in enumerate(varargs):
            _check_type(t)
            if isinstance(t, BitField):
                raise InvalidTypeError(
                    "a bare BitField cannot be a variadic argument",
                    f"vararg[{i}]")
            vnorm.append(t)
        if ret is not None:
            _check_type(ret)
            if isinstance(ret, BitField):
                raise InvalidTypeError("a bare BitField cannot be a return type")
        names: set[str] = set()
        cnorm = []
        for item in cells:
            if not (isinstance(item, tuple) and len(item) == 2):
                raise InvalidTypeError(f"cell must be (name, Type), got {item!r}")
            cname, ctype = item
            if not isinstance(cname, str) or not cname:
                raise InvalidTypeError(f"bad cell name {cname!r}")
            if cname in names:
                raise InvalidTypeError(f"duplicate cell name {cname!r}")
            names.add(cname)
            _check_type(ctype)
            cnorm.append((cname, ctype))
        if ret_cell is not None:
            if ret_cell not in names:
                raise InvalidTypeError(
                    f"ret_cell {ret_cell!r} is not declared in cells")
            if ret is not None:
                cell_type = dict(cnorm)[ret_cell]
                if cell_type is not ret:
                    raise InvalidTypeError(
                        f"ret_cell {ret_cell!r} type {cell_type.name} is not "
                        f"the same Type object as ret {ret.name}")
        object.__setattr__(self, "args", tuple(norm))
        object.__setattr__(self, "varargs", tuple(vnorm))
        object.__setattr__(self, "ret", ret)
        object.__setattr__(self, "cells", tuple(cnorm))
        object.__setattr__(self, "ret_cell", ret_cell)


@dataclass(frozen=True)
class V2Placement:
    """一个逻辑实参的安放结果（v2）。"""

    kind: str                        # "registers" / "stack" / "none"
    eightbytes: tuple[str, ...] = ()
    int_regs: tuple[int, ...] = ()
    float_regs: tuple[int, ...] = ()
    frame_offset: int | None = None
    size: int = 0
    promoted_from: Type | None = None
    path: str = "arg[0]"


@dataclass(frozen=True)
class CellInfo:
    name: str
    type: Type
    layout: Layout
    offset: int


@dataclass(frozen=True)
class ReturnInfo:
    kind: str                        # "none" / "reg" / "sret" / "cell"
    eightbytes: tuple[str, ...] = ()
    cell: str | None = None
    hidden_arg: V2Placement | None = None


@dataclass(frozen=True)
class SaveArea:
    """可变参数寄存器保存区（帧内两段按位拷贝）。"""

    int_offset: int
    int_slots: tuple[int, ...]
    float_offset: int
    float_slots: tuple[int, ...]
    size: int


@dataclass(frozen=True)
class Plan:
    descriptor: AbiDescriptor
    signature: Signature
    layouts: tuple[Layout, ...]
    vararg_layouts: tuple[Layout, ...]
    promoted: tuple[Type | None, ...]
    placements: tuple[V2Placement, ...]
    vararg_placements: tuple[V2Placement, ...]
    frame_size: int
    cells: tuple[CellInfo, ...]
    ret: ReturnInfo
    save: SaveArea | None
    certificate: "Certificate"
    work: int
    budget: int
    status: str = "ok"


@dataclass(frozen=True)
class BitEvidence:
    """逐位证据：region 的 [start, start+length) 位的来源。

    region: "ireg:k" / "freg:k" / "frame"（位编号相对该区域起点）。
    kind:
      "value"    参数/字段值的位（path 给出参数路径，src_bit 给值内位索引；
                 src_bit=-1 表示按自然位序）
      "tag"      联合体标签位
      "union"    联合体共享存储位（运行时由 tag 选择活动成员；标签在同路径
                 的 tag 证据中）
      "save-copy" 可变参数保存区，源是某寄存器槽（src_region）
      "cell"     共享单元内容
      "ret"      返回值载体（sret 单元 / ret_cell）占的位
      "pad"      确定性零填充（含窄标量入槽的高位——两描述符一律零扩展，
                 有符号性由类型保留，不靠槽内符号扩展位）
    """

    region: str
    start: int
    length: int
    kind: str
    path: str = ""
    src_bit: int = -1
    src_region: str = ""
    shared_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class Certificate:
    entries: tuple[BitEvidence, ...]
    frame_bits: int
    int_regs: int
    float_regs: int


def check_certificate(cert: Certificate) -> list[str]:
    """证书自检：每寄存器槽 64 位、帧 frame_bits 位被恰好覆盖一次。

    返回问题列表（空 = 通过）。这是规划器自带的一致性检查；验收方的独立
    检查不依赖本函数结论，而是从类型与描述符重新推导。
    """
    issues: list[str] = []
    covered: dict[str, int] = {}
    totals = {"frame": cert.frame_bits}
    for k in range(cert.int_regs):
        totals[f"ireg:{k}"] = 64
    for k in range(cert.float_regs):
        totals[f"freg:{k}"] = 64

    for e in cert.entries:
        total = totals.get(e.region)
        if total is None:
            issues.append(f"unknown region {e.region!r}")
            continue
        if e.start < 0 or e.length <= 0 or e.start + e.length > total:
            issues.append(f"{e.region}: range [{e.start},{e.start + e.length}) "
                          f"out of bounds [0,{total})")
            continue
        bits = ((1 << e.length) - 1) << e.start
        mask = covered.setdefault(e.region, 0)
        if mask & bits:
            issues.append(f"{e.region}: overlapping coverage at "
                          f"[{e.start},{e.start + e.length}) ({e.kind})")
        covered[e.region] = mask | bits

    full = {region: ((1 << total) - 1 if total else 0)
            for region, total in totals.items()}
    for region, want in full.items():
        if covered.get(region, 0) != want:
            miss = want & ~covered.get(region, 0)
            extra = covered.get(region, 0) & ~want
            issues.append(f"{region}: incomplete coverage "
                          f"(uncovered={miss.bit_count() if miss else 0}, "
                          f"out-of-range={extra.bit_count() if extra else 0})")
    return issues


def _promote(desc: AbiDescriptor, t: Type, path: str) -> Type:
    """可变参数默认提升（v2 明示规则）：
    char/short（<4 字节整数）→ 同符号性 32 位整数；f32 → f64；
    128 位整数无提升目标 → PromotionError。
    """
    if isinstance(t, IntType) and t.size < 4:
        return IntType(4, t.signed)
    if isinstance(t, Float32Type):
        return FloatType()
    if isinstance(t, IntNType):
        raise PromotionError(
            f"{t.name} cannot be passed as a variadic argument: the default "
            "argument promotions have no representable target for 128-bit "
            "integers in this descriptor", path)
    return t


def plan_call(desc: AbiDescriptor, signature: Signature,
              *, budget: int = DEFAULT_BUDGET) -> Plan:
    """纯静态规划：不接触任何值，输出安放方案与逐位证书。"""
    if not isinstance(desc, AbiDescriptor):
        raise InvalidTypeError(f"expected an AbiDescriptor, got {desc!r}")
    if not isinstance(signature, Signature):
        raise InvalidTypeError(f"expected a Signature, got {signature!r}")
    if len(signature.args) > desc.max_args:
        raise InvalidTypeError(
            f"too many named arguments: {len(signature.args)} > "
            f"{desc.max_args}", "args")
    if len(signature.varargs) > desc.max_varargs:
        raise InvalidTypeError(
            f"too many variadic arguments: {len(signature.varargs)} > "
            f"{desc.max_varargs}", "varargs")

    work = 0

    def charge(bits: int):
        nonlocal work
        work += max(bits, 1)
        if work > budget:
            raise BudgetExhausted(
                f"planner budget exhausted: {work} > {budget} work units")

    def lay(t, p):
        try:
            return _layout_v2(desc, t)
        except BitFieldError as exc:
            raise BitFieldError(
                exc.message,
                f"{p}: {exc.path}" if exc.path else p) from None

    layouts = tuple(lay(t, f"arg[{i}]")
                    for i, t in enumerate(signature.args))
    promoted: list[Type | None] = []
    vtypes: list[Type] = []
    for i, t in enumerate(signature.varargs):
        pt = _promote(desc, t, f"vararg[{i}]")
        promoted.append(pt if pt is not t else None)
        vtypes.append(pt)
    vararg_layouts = tuple(lay(t, f"vararg[{i}]")
                           for i, t in enumerate(vtypes))
    ret_lay = lay(signature.ret, "ret") if signature.ret else None
    for L in layouts + vararg_layouts:
        charge(L.size * 8 + 64)
    if ret_lay:
        charge(ret_lay.size * 8 + 64)

    # ---- 返回值策略 ----
    if signature.ret is None:
        ret_info = ReturnInfo("none")
    elif signature.ret_cell is not None:
        ret_info = ReturnInfo("cell", (), signature.ret_cell)
    else:
        cls = classify(signature.ret, abi=desc)
        if ret_lay.size > desc.sret_threshold or cls == (MEMORY,):
            ret_info = ReturnInfo("sret", (MEMORY,), "$sret")
        else:
            ret_info = ReturnInfo("reg", cls)

    # ---- 帧与寄存器分配 ----
    frame_len = 0
    next_int = 0
    next_float = 0
    placements: list[V2Placement] = []
    vplacements: list[V2Placement] = []

    hidden = None
    if ret_info.kind == "sret":
        hidden, frame_len, next_int, next_float = _allocate(
            desc, PTR, _layout_v2(desc, PTR), "$sret.hidden",
            frame_len, next_int, next_float)
        ret_info = ReturnInfo("sret", (MEMORY,), "$sret", hidden)

    for i, (t, L) in enumerate(zip(signature.args, layouts)):
        pl, frame_len, next_int, next_float = _allocate(
            desc, t, L, f"arg[{i}]", frame_len, next_int, next_float)
        placements.append(pl)

    save_int_slots: list[int] = []
    save_float_slots: list[int] = []
    for i, (t, L, pt) in enumerate(
            zip(vtypes, vararg_layouts, promoted)):
        pl, frame_len, next_int, next_float = _allocate(
            desc, t, L, f"vararg[{i}]", frame_len, next_int, next_float,
            promoted_from=pt)
        vplacements.append(pl)
        if pl.kind == "registers":
            save_int_slots.extend(pl.int_regs)
            save_float_slots.extend(pl.float_regs)

    # ---- 共享单元区（声明单元 + 隐式 $sret）----
    cells: list[CellInfo] = []
    cursor = _align_up(frame_len, desc.stack_align)
    for cname, ctype in signature.cells:
        cl = _layout_v2(desc, ctype)
        al = min(max(cl.align, desc.stack_align), desc.max_agg_align)
        cursor = _align_up(cursor, al)
        cells.append(CellInfo(cname, ctype, cl, cursor))
        charge(cl.size * 8)
        cursor += cl.size
    if ret_info.kind == "sret":
        al = min(max(ret_lay.align, desc.stack_align), desc.max_agg_align)
        cursor = _align_up(cursor, al)
        cells.append(CellInfo("$sret", signature.ret, ret_lay, cursor))
        charge(ret_lay.size * 8)
        cursor += ret_lay.size

    # ---- 可变参数寄存器保存区 ----
    save = None
    if save_int_slots or save_float_slots:
        int_off = _align_up(cursor, desc.stack_align)
        cursor = int_off + 8 * len(save_int_slots)
        float_off = _align_up(cursor, desc.stack_align)
        cursor = float_off + 8 * len(save_float_slots)
        save = SaveArea(int_off, tuple(save_int_slots), float_off,
                        tuple(save_float_slots), cursor - int_off)
        charge(save.size * 8)
    frame_total = _align_up(cursor, desc.stack_align)
    charge(frame_total * 8)

    cert = _build_certificate(
        desc, signature, layouts, vararg_layouts, vtypes,
        placements, vplacements, cells, ret_info, ret_lay, save, frame_total)
    issues = check_certificate(cert)
    if issues:  # 防御性：规划器内部不变量被破坏
        raise ABIError("certificate invariant failure: " + "; ".join(issues))

    return Plan(desc, signature, tuple(layouts), tuple(vararg_layouts),
                tuple(promoted), tuple(placements), tuple(vplacements),
                frame_total, tuple(cells), ret_info, save, cert,
                work, budget, "ok")


def _allocate(desc, t, lay, path, frame_len, next_int, next_float, *,
              promoted_from=None):
    """分配一个实参。整体回退：寄存器不足（含 i128 配对落空）时整体走栈，
    不消耗任何寄存器。"""
    cls = classify(t, abi=desc)
    if not cls:
        return (V2Placement("none", (), (), (), None, 0, promoted_from, path),
                frame_len, next_int, next_float)
    if cls == (MEMORY,):
        off = _align_up(frame_len, desc.stack_align)
        return (V2Placement("stack", cls, (), (), off, lay.size,
                            promoted_from, path),
                off + lay.size, next_int, next_float)

    need_int = cls.count(INTEGER)
    need_float = cls.count(FLOAT)
    start_int = next_int
    if isinstance(t, IntNType) and desc.i128_even_pair and next_int % 2 == 1:
        start_int = next_int + 1  # 奇数槽留空，不回溯
    if (start_int + need_int <= desc.int_reg_count
            and next_float + need_float <= desc.float_reg_count):
        return (V2Placement(
            "registers", cls,
            tuple(range(start_int, start_int + need_int)),
            tuple(range(next_float, next_float + need_float)),
            None, lay.size, promoted_from, path),
                frame_len, start_int + need_int,
                next_float + need_float)

    off = _align_up(frame_len, desc.stack_align)
    return (V2Placement("stack", cls, (), (), off, lay.size,
                        promoted_from, path),
            off + lay.size, next_int, next_float)


# ===========================================================================
# v2：逐位证书构造
# ===========================================================================

def _union_storage_size(lay: Layout) -> int:
    return max((m.layout.size for m in lay.members), default=0)


def _value_ranges(desc: AbiDescriptor, t: Type, lay: Layout, base_bit: int,
                  path: str, out: list) -> None:
    """登记类型内全部来源位区间（位粒度，位置已是该描述符下的物理位置）。

    每项 (start_bit, length, kind, path, src_bit)；联合体的共享存储登记为
    单段 "union"（运行时由 tag 选择活动成员，标签位另计）。
    """
    if isinstance(t, (IntType, IntNType, PtrType, FloatType, Float32Type)):
        out.append((base_bit, lay.size * 8, "value", path, -1))
    elif isinstance(t, ArrayType):
        step = lay.size // t.count if t.count else 0
        elay = _layout_v2(desc, t.elem)
        for i in range(t.count):
            _value_ranges(desc, t.elem, elay,
                          base_bit + i * step * 8, f"{path}[{i}]", out)
    elif isinstance(t, RecordType):
        for f in lay.fields:
            if f.bit_width is not None:
                # 逐物理位登记（大端段内可能反向，见 BitSegment.rev）
                for seg in f.bit_segments:
                    for k in range(seg.count):
                        vb = seg.value_bit + (
                            seg.count - 1 - k if seg.rev else k)
                        out.append((
                            base_bit + (f.offset + seg.byte) * 8
                            + seg.lsb + k,
                            1, "value", f"{path}.{f.name}", vb))
            else:
                _value_ranges(desc, f.type, f.layout,
                              base_bit + f.offset * 8, f"{path}.{f.name}", out)
    elif isinstance(t, UnionType):
        out.append((base_bit + lay.tag_offset * 8,
                    lay.tag_layout.size * 8, "tag", path, -1))
        out.append((base_bit + lay.storage_offset * 8,
                    _union_storage_size(lay) * 8, "union", path, -1))
    else:  # pragma: no cover
        raise InvalidTypeError(f"unsupported type {t!r}")


def _build_certificate(desc, sig, layouts, vararg_layouts, vtypes,
                       placements, vplacements, cells, ret_info,
                       ret_lay, save, frame_total) -> Certificate:
    entries: list[BitEvidence] = []
    used_ireg: set[int] = set()
    used_freg: set[int] = set()
    frame_ev: list[BitEvidence] = []

    def mark_frame(start, length, kind, path="", src_bit=-1,
                   src_region="", shared_by=()):
        frame_ev.append(BitEvidence("frame", start, length, kind, path,
                                    src_bit, src_region, tuple(shared_by)))

    def cover_stack(pl, t, lay, path):
        ranges: list = []
        _value_ranges(desc, t, lay, pl.frame_offset * 8, path, ranges)
        for start, length, kind, p, sb in ranges:
            mark_frame(start, length, kind, p, sb)

    def cover_regs(pl, t, lay, path):
        ranges: list = []
        _value_ranges(desc, t, lay, 0, path, ranges)
        # 按 8 字节块切分（标量可能跨块边界）
        per_block: dict[int, list] = {}
        for start, length, kind, p, sb in ranges:
            remain = length
            cur = start
            vsb = sb
            while remain:
                block = cur // 64
                cap = 64 - cur % 64
                take = min(remain, cap)
                per_block.setdefault(block, []).append(
                    (cur % 64, take, kind, p, vsb))
                cur += take
                remain -= take
                if sb >= 0:
                    vsb += take
        ii = iter(pl.int_regs)
        fi = iter(pl.float_regs)
        for bi, c in enumerate(pl.eightbytes):
            slot = next(ii) if c == INTEGER else next(fi)
            region = f"{'ireg' if c == INTEGER else 'freg'}:{slot}"
            if c == INTEGER:
                used_ireg.add(slot)
            else:
                used_freg.add(slot)
            covered_mask = 0
            for lsb, count, kind, p, sb in per_block.get(bi, []):
                _emit_slot_runs(desc, entries, region, lsb, count, kind, p, sb,
                                mask_holder=(covered_mask,))
                covered_mask |= _slot_bits(desc, lsb, count)
            # 槽内未覆盖位 → pad（窄标量零扩展约定）
            _emit_pad_runs(entries, region, covered_mask)

    def cover(pl, t, lay, path):
        if pl.kind == "none":
            return
        if pl.kind == "stack":
            cover_stack(pl, t, lay, path)
        else:
            cover_regs(pl, t, lay, path)

    # 隐式 sret 隐藏指针
    h = ret_info.hidden_arg
    if h is not None:
        if h.kind == "stack":
            mark_frame(h.frame_offset * 8, 64, "value", "$sret.hidden")
        elif h.kind == "registers":
            used_ireg.add(h.int_regs[0])
            entries.append(BitEvidence(f"ireg:{h.int_regs[0]}", 0, 64,
                                       "value", "$sret.hidden"))

    for i, (pl, t, L) in enumerate(zip(placements, sig.args, layouts)):
        cover(pl, t, L, f"arg[{i}]")
    for i, (pl, t, L) in enumerate(zip(vplacements, vtypes, vararg_layouts)):
        cover(pl, t, L, f"vararg[{i}]")

    # 单元区
    for cell in cells:
        readers = ["ret"] if (cell.name == "$sret"
                              or sig.ret_cell == cell.name) else []
        kind = "ret" if readers else "cell"
        ranges: list = []
        cpath = f"cell:{cell.name}"
        _value_ranges(desc, cell.type, cell.layout, cell.offset * 8, cpath,
                      ranges)
        for start, length, k, p, sb in ranges:
            mark_frame(start, length,
                       "tag" if k == "tag" else
                       ("union" if k == "union" else kind),
                       p, sb, shared_by=readers)

    # 保存区（按位拷贝；内存映像字节序由描述符决定）
    if save is not None:
        for j, slot in enumerate(save.int_slots):
            start = (save.int_offset + 8 * j) * 8
            mark_frame(start, 64, "save-copy", f"ireg:{slot}",
                       src_region=f"ireg:{slot}")
        for j, slot in enumerate(save.float_slots):
            start = (save.float_offset + 8 * j) * 8
            mark_frame(start, 64, "save-copy", f"freg:{slot}",
                       src_region=f"freg:{slot}")

    # 未使用寄存器槽
    for k in range(desc.int_reg_count):
        if k not in used_ireg:
            entries.append(BitEvidence(f"ireg:{k}", 0, 64, "pad"))
    for k in range(desc.float_reg_count):
        if k not in used_freg:
            entries.append(BitEvidence(f"freg:{k}", 0, 64, "pad"))

    # 帧间隙/尾部填零（区间排序后线性扫描）
    frame_ev.sort(key=lambda e: (e.start, e.length))
    cursor = 0
    for ev in frame_ev:
        if ev.start > cursor:
            entries.append(BitEvidence("frame", cursor, ev.start - cursor,
                                       "pad"))
        entries.append(ev)
        cursor = ev.start + ev.length
    if cursor < frame_total * 8:
        entries.append(BitEvidence("frame", cursor, frame_total * 8 - cursor,
                                   "pad"))
    return Certificate(tuple(entries), frame_total * 8,
                       desc.int_reg_count, desc.float_reg_count)


def _slot_phy_bits(desc: AbiDescriptor, data_lsb: int, count: int) -> list[int]:
    """块内位地址（相对 8 字节块起点）→ 槽内物理位编号。

    槽是块的 8 字节内存别名：槽位 = 块内字节*8 + 字节内位，与字节序无关
    （字节序对字段/位域物理位置的影响已在类型布局层处理）。
    """
    return [data_lsb + b for b in range(count)]


def _slot_bits(desc: AbiDescriptor, data_lsb: int, count: int) -> int:
    m = 0
    for b in _slot_phy_bits(desc, data_lsb, count):
        m |= 1 << b
    return m


def _emit_slot_runs(desc, entries, region, data_lsb, count, kind, path,
                    src_bit, mask_holder):
    """把一段连续数据位写入证书（物理位可能在字节交换下保持连续）。"""
    phy = _slot_phy_bits(desc, data_lsb, count)
    # 按物理位排序后合并连续段
    groups: list[tuple[int, int, int]] = []  # (phy_lsb, n, first_src_index)
    for order, pb in enumerate(phy):
        if groups and pb == groups[-1][0] + groups[-1][1]:
            g = groups[-1]
            groups[-1] = (g[0], g[1] + 1, g[2])
        else:
            groups.append((pb, 1, order))
    for phy_lsb, n, order in groups:
        sb = src_bit + order if src_bit >= 0 else -1
        entries.append(BitEvidence(region, phy_lsb, n, kind, path, sb))


def _emit_pad_runs(entries, region, covered_mask: int):
    remain = ((1 << 64) - 1) & ~covered_mask
    while remain:
        b = (remain & -remain).bit_length() - 1
        run = 0
        while b + run < 64 and (remain >> (b + run)) & 1:
            run += 1
        entries.append(BitEvidence(region, b, run, "pad"))
        remain &= ~(((1 << run) - 1) << b)


# ===========================================================================
# v2：调用编码 / 解码
# ===========================================================================

@dataclass(frozen=True)
class V2CallImage:
    """v2 调用映像。

    int_regs/float_regs：每槽 None 或原始 64 位无符号整数。浮点槽同样以
    原始位模式存放（NaN 载荷无损）。frame：栈参数 + 共享单元 + 可变参数
    保存区拼接的帧字节。ret_int/ret_float：寄存器返回值槽的原始位，
    按返回分类顺序排列。
    """

    int_regs: tuple
    float_regs: tuple
    frame: bytes
    placements: tuple[V2Placement, ...]
    vararg_placements: tuple[V2Placement, ...]
    ret_int: tuple
    ret_float: tuple
    plan: Plan


@dataclass(frozen=True)
class CallValues:
    """decode_call 的规范值结果。"""

    args: list
    varargs: list
    cells: dict
    ret: object


_MISSING = object()


def _slot_raw(desc: AbiDescriptor, chunk: bytes) -> int:
    """把块内 n<=8 字节数据放入 64 位槽。

    槽语义（两种字节序统一）：槽就是该 8 字节块的内存别名，块数据放在槽
    的低 n 字节（块内自然偏移），其余字节为零；返回的整数按**物理 little
    位序**编号（与描述符字节序无关）。标量的字节序含义只体现在
    encode/decode 与位域物理段上，不改变块在槽中的位置。
    """
    buf = bytearray(8)
    buf[:len(chunk)] = chunk
    return int.from_bytes(buf, "little")


def _slot_extract(raw: int, desc: AbiDescriptor, n: int) -> bytes:
    """_slot_raw 的逆：取回槽低 n 个块字节。"""
    buf = raw.to_bytes(8, "little")
    return buf if n == 8 else buf[:n]


def encode_call(plan: Plan, arg_values, vararg_values=(),
                cell_values=None, ret_value=_MISSING) -> V2CallImage:
    """按已完成的 plan 编码一次调用。

    所有值先校验并写入临时缓冲，全部成功后才构造不可变 V2CallImage，
    不存在部分结果。f32 可变参数先舍入到 f32 再提升为 f64，其规范值域
    为“可由 binary32 精确表示的 float”；NaN 经 f32 域保留其 f32 载荷。
    """
    if not isinstance(plan, Plan):
        raise InvalidTypeError(f"expected a Plan, got {plan!r}")
    desc = plan.descriptor
    sig = plan.signature
    arg_values = list(arg_values)
    vararg_values = list(vararg_values)
    cell_values = dict(cell_values or {})
    if len(arg_values) != len(sig.args):
        raise InvalidValueError(
            f"expected {len(sig.args)} named values, got {len(arg_values)}",
            "args")
    if len(vararg_values) != len(sig.varargs):
        raise InvalidValueError(
            f"expected {len(sig.varargs)} variadic values, got "
            f"{len(vararg_values)}", "varargs")
    required_cells = {n for n, _ in sig.cells if sig.ret_cell != n}
    missing = required_cells - set(cell_values)
    if missing:
        raise InvalidValueError(f"missing cell values: {sorted(missing)}",
                                "cells")

    int_regs: list = [None] * desc.int_reg_count
    float_regs: list = [None] * desc.float_reg_count
    frame = bytearray(plan.frame_size)
    off_of = {c.name: c.offset for c in plan.cells}

    def resolve_ptr(v, path):
        if isinstance(v, CellPtr):
            if v.cell not in off_of:
                raise InvalidValueError(f"unknown cell {v.cell!r}", path)
            return FRAME_BASE + off_of[v.cell]
        return v

    # ---- 共享单元（ret_cell / $sret 承载返回值）----
    for cname, ctype in sig.cells:
        if sig.ret_cell == cname:
            if ret_value is _MISSING:
                raise InvalidValueError(
                    "signature returns via ret_cell; ret_value required",
                    "ret")
            val = ret_value
            cpath = "ret"
        else:
            val = cell_values[cname]
            cpath = f"cell:{cname}"
        cl = next(c.layout for c in plan.cells if c.name == cname)
        data = _encode_v2(desc, ctype, val, cl, cpath)
        off = off_of[cname]
        frame[off:off + len(data)] = data
    if plan.ret.kind == "sret":
        if ret_value is _MISSING:
            raise InvalidValueError(
                "signature has a return value; ret_value is required", "ret")
        c = next(c for c in plan.cells if c.name == "$sret")
        frame[c.offset:c.offset + c.layout.size] = \
            _encode_v2(desc, sig.ret, ret_value, c.layout, "ret")

    def emit(pl, t, lay, value, path):
        if pl.kind == "none":
            # 空聚合仍做一次值校验
            _encode_v2(desc, t, value, lay, path)
            return
        if isinstance(t, PtrType):
            value = resolve_ptr(value, path)
        data = _encode_v2(desc, t, value, lay, path)
        if pl.kind == "stack":
            frame[pl.frame_offset:pl.frame_offset + len(data)] = data
            return
        ii = iter(pl.int_regs)
        fi = iter(pl.float_regs)
        n_full = len(data) // 8
        tail = len(data) % 8
        for k, c in enumerate(pl.eightbytes):
            if k < n_full:
                chunk = data[8 * k:8 * k + 8]
            else:
                chunk = data[8 * k:8 * k + tail]
            if c == INTEGER:
                int_regs[next(ii)] = _slot_raw(desc, chunk)
            else:
                float_regs[next(fi)] = _slot_raw(desc, chunk)

    # 隐式 sret 指针
    h = plan.ret.hidden_arg
    if h is not None:
        addr = FRAME_BASE + off_of["$sret"]
        data = addr.to_bytes(8, desc.endian)
        if h.kind == "stack":
            frame[h.frame_offset:h.frame_offset + 8] = data
        else:
            int_regs[h.int_regs[0]] = _slot_raw(desc, data)

    for i, ((t, L), pl, v) in enumerate(
            zip(zip(sig.args, plan.layouts), plan.placements, arg_values)):
        emit(pl, t, L, v, f"arg[{i}]")

    for i, (t0, pl, v) in enumerate(
            zip(sig.varargs, plan.vararg_placements, vararg_values)):
        tt = plan.promoted[i] or t0
        if isinstance(t0, IntType) and t0.size < 4:
            _check_int_value(v, t0.size, t0.signed, f"vararg[{i}]", t0.name)
        elif isinstance(t0, Float32Type):
            # 先舍入到 f32 规范值域，再提升为 f64
            v = struct.unpack(desc.endian_char + "f",
                              struct.pack(desc.endian_char + "f", v))[0]
        emit(pl, tt, plan.vararg_layouts[i], v, f"vararg[{i}]")

    # 保存区：寄存器槽原始 64 位的按位拷贝。保存区定义为与数据字节序无关
    # 的位映像（槽位 j -> 帧 base+j 位），两种描述符下逐位相同。
    if plan.save is not None:
        sv = plan.save
        for j, slot in enumerate(sv.int_slots):
            frame[sv.int_offset + 8 * j:sv.int_offset + 8 * j + 8] = \
                int_regs[slot].to_bytes(8, "little")
        for j, slot in enumerate(sv.float_slots):
            frame[sv.float_offset + 8 * j:sv.float_offset + 8 * j + 8] = \
                float_regs[slot].to_bytes(8, "little")

    ret_int: list = []
    ret_float: list = []
    if plan.ret.kind == "reg":
        if ret_value is _MISSING:
            raise InvalidValueError(
                "signature has a return value; ret_value is required", "ret")
        rlay = _layout_v2(desc, sig.ret)
        data = _encode_v2(desc, sig.ret, ret_value, rlay, "ret")
        for k, c in enumerate(plan.ret.eightbytes):
            chunk = data[8 * k:8 * k + 8] or b"\x00" * 8
            (ret_int if c == INTEGER else ret_float).append(_slot_raw(desc, chunk))

    image = V2CallImage(
        tuple(int_regs), tuple(float_regs), bytes(frame),
        plan.placements, plan.vararg_placements,
        tuple(ret_int), tuple(ret_float), plan)

    # 编码后不变量：证书声明为非 pad 的寄存器槽必须有值
    non_pad: set[str] = set()
    for e in plan.certificate.entries:
        if e.kind != "pad":
            non_pad.add(e.region)
    for k in range(desc.int_reg_count):
        if f"ireg:{k}" in non_pad and image.int_regs[k] is None:
            raise ABIError(f"internal: ireg:{k} certified but empty")
    for k in range(desc.float_reg_count):
        if f"freg:{k}" in non_pad and image.float_regs[k] is None:
            raise ABIError(f"internal: freg:{k} certified but empty")
    return image


def decode_call(plan: Plan, image: V2CallImage) -> CallValues:
    """从映像按 plan 重建规范值（联合体按标签选择活动成员）。"""
    if not isinstance(plan, Plan):
        raise InvalidTypeError(f"expected a Plan, got {plan!r}")
    if not isinstance(image, V2CallImage):
        raise InvalidValueError(f"expected a V2CallImage, got {image!r}")
    desc = plan.descriptor
    sig = plan.signature

    def read(pl, t, lay, path):
        if pl.kind == "none":
            return _decode_v2(desc, t, bytes(lay.size), lay, path)
        if pl.kind == "stack":
            data = image.frame[pl.frame_offset:pl.frame_offset + pl.size]
            if len(data) != pl.size:
                raise InvalidValueError(
                    f"frame truncated: need {pl.size} bytes at "
                    f"{pl.frame_offset}, have {len(data)}", path)
        else:
            chunks: list[bytes] = []
            ii = iter(pl.int_regs)
            fi = iter(pl.float_regs)
            n_full = lay.size // 8
            tail = lay.size % 8
            for k, c in enumerate(pl.eightbytes):
                raw = image.int_regs[next(ii)] if c == INTEGER \
                    else image.float_regs[next(fi)]
                if raw is None:
                    raise InvalidValueError(
                        f"register slot for {path} is empty", path)
                if k < n_full:
                    chunks.append(_slot_extract(raw, desc, 8))
                elif k == n_full and tail:
                    chunks.append(_slot_extract(raw, desc, tail))
            data = b"".join(chunks)
        return _decode_v2(desc, t, data, lay, path)

    args = []
    for i, (t, L, pl) in enumerate(
            zip(sig.args, plan.layouts, plan.placements)):
        v = read(pl, t, L, f"arg[{i}]")
        if isinstance(t, PtrType) and isinstance(v, int):
            v = _rebase_pointer(plan, v)
        args.append(v)

    varargs = []
    for i, (t0, pl) in enumerate(zip(sig.varargs, plan.vararg_placements)):
        tt = plan.promoted[i] or t0
        v = read(pl, tt, plan.vararg_layouts[i], f"vararg[{i}]")
        varargs.append(_demote(desc, t0, tt, v, f"vararg[{i}]"))

    cells = {}
    for c in plan.cells:
        if c.name == "$sret":
            continue
        data = image.frame[c.offset:c.offset + c.layout.size]
        val = _decode_v2(desc, c.type, data, c.layout, f"cell:{c.name}")
        cells[c.name] = val

    ret = None
    if plan.ret.kind == "reg":
        chunks: list[bytes] = []
        ii = iter(image.ret_int)
        fi = iter(image.ret_float)
        rlay = _layout_v2(desc, sig.ret)
        n_full = rlay.size // 8
        tail = rlay.size % 8
        for k, c in enumerate(plan.ret.eightbytes):
            raw = next(ii) if c == INTEGER else next(fi)
            if k < n_full:
                chunks.append(_slot_extract(raw, desc, 8))
            elif k == n_full and tail:
                chunks.append(_slot_extract(raw, desc, tail))
        ret = _decode_v2(desc, sig.ret, b"".join(chunks), rlay, "ret")
    elif plan.ret.kind in ("sret", "cell"):
        cname = plan.ret.cell
        c = next(c for c in plan.cells if c.name == cname)
        data = image.frame[c.offset:c.offset + c.layout.size]
        ret = _decode_v2(desc, sig.ret, data, c.layout, "ret")

    return CallValues(args, varargs, cells, ret)


def _rebase_pointer(plan: Plan, addr: int):
    for c in plan.cells:
        if addr == FRAME_BASE + c.offset:
            return CellPtr(c.name)
    return addr


def _demote(desc, original: Type, promoted: Type, value, path: str):
    """提升后的规范值还原为原可变参数类型的值。"""
    if isinstance(original, Float32Type) and isinstance(promoted, FloatType):
        # value 已是 f32 精确值经 f64 承载；按 f32 还原（NaN 载荷随 f32 域）。
        return struct.unpack(desc.endian_char + "f",
                             struct.pack(desc.endian_char + "f", value))[0]
    if isinstance(original, IntType) and original.size < 4 \
            and isinstance(promoted, IntType):
        lo, hi = original.range
        if not lo <= value <= hi:
            raise PromotionError(
                f"promoted value {value} does not demote to {original.name}",
                path)
        return value
    return value


# ===========================================================================
# v2：布局 A <-> 布局 B 桥接
# ===========================================================================

def bridge_call(src_plan: Plan, src_image: V2CallImage,
                dst_desc: AbiDescriptor, *,
                budget: int = DEFAULT_BUDGET) -> V2CallImage:
    """把源布局映像转换为目标描述符下的映像。

    先 decode_call 得规范值，再按 dst_desc 规划并编码；失败在写入任何目标
    状态前抛 BridgeError（reason: value/promotion/tag/unidentifiable/budget）。
    """
    try:
        decoded = decode_call(src_plan, src_image)
    except TagError as exc:
        raise BridgeError(exc.message, "tag", exc.path) from None
    except PromotionError as exc:
        raise BridgeError(exc.message, "promotion", exc.path) from None
    except ABIError as exc:
        raise BridgeError(exc.message, "value", exc.path) from None

    sig = src_plan.signature
    try:
        dst_plan = plan_call(dst_desc, sig, budget=budget)
    except BudgetExhausted as exc:
        raise BridgeError(str(exc), "budget") from None
    except ABIError as exc:
        raise BridgeError(exc.message, "unidentifiable", exc.path) from None

    try:
        return encode_call(dst_plan, decoded.args, decoded.varargs,
                           decoded.cells, decoded.ret)
    except TagError as exc:
        raise BridgeError(exc.message, "tag", exc.path) from None
    except PromotionError as exc:
        raise BridgeError(exc.message, "promotion", exc.path) from None
    except IntegerOverflowError as exc:
        raise BridgeError(exc.message, "value", exc.path) from None
    except ABIError as exc:
        raise BridgeError(exc.message, "value",
                          getattr(exc, "path", None)) from None


def bridge_into(src_plan: Plan, src_image: V2CallImage,
                dst_desc: AbiDescriptor, buffer: bytearray,
                *, budget: int = DEFAULT_BUDGET) -> tuple[str, Plan | None]:
    """桥接并把目标帧写入调用方提供的 buffer。

    返回 ("ok", dst_plan) 或 ("rejected", None)；失败时 buffer 不发生任何
    写入（先在临时映像中构造成功后才整体拷贝）。
    """
    if not isinstance(buffer, bytearray):
        raise InvalidTypeError("buffer must be a bytearray")
    try:
        dst_image = bridge_call(src_plan, src_image, dst_desc, budget=budget)
    except BridgeError:
        return ("rejected", None)
    buffer[:] = dst_image.frame
    return ("ok", dst_image.plan)
