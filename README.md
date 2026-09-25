# abi_marshal — 聚合参数与返回值的合成 ABI 编组

离线外部函数接口工程的可复用模块：把 `(类型, 值)` 实参序列编组为
“寄存器 + 栈”的调用映像，把返回值编组为“返回寄存器”或“有界缓冲区
间接返回”，并可从映像重建参数与返回值。

## 运行环境

- Python **3.14.7**，仅标准库，**无需安装任何额外依赖**；
- Linux 原生运行；核心、测试、演示全部离线，不需要网络、账号、密钥、
  数据库、中间件、云 API 或 Docker。

## 命令

```sh
./.venv/bin/python -m unittest discover -s tests -v   # 运行测试（70 个用例）
./.venv/bin/python demo.py                            # 固定输入演示（约 1 秒内完成）
```

## 明示 ABI 契约（本模块实现且仅实现此子集）

- 小端字节序；指针 8 字节（无符号）；
- 整数 1/2/4/8 字节（有/无符号）；浮点 8 字节（IEEE-754 binary64）；
- 记录按自然对齐布局，对齐上限 8；空记录 `size=0, align=1`；
- 数组为显式长度的同类元素序列，元素按完整 size 连续排列；
- 聚合按 8 字节块分类：块内含任意整数/指针字节 → `INTEGER`，否则
  `FLOAT`；聚合超过 16 字节 → `MEMORY`；
- **参数**：整数寄存器 4 个、浮点寄存器 2 个；聚合所需寄存器不足时
  **整体**走 16 字节对齐的栈，且**不消耗任何寄存器**（不允许拆半安放）；
  空记录不消耗寄存器也不消耗栈；
- **返回**：标量及不超过 16 字节的聚合沿用同一块分类，逐块进入返回
  寄存器——整数返回寄存器 2 个、浮点返回寄存器 2 个；空记录不占任何
  返回寄存器；
- **返回**：超过 16 字节的聚合由调用方传入**有界返回缓冲区**间接返回，
  被调方把完整布局（padding 一律写零）一次性写入该缓冲区；缓冲区的
  容量、对齐与地址范围**先验证**，任何失败都不得写入部分结果；
- **交互**：间接返回的隐藏指针**先占一个整数参数寄存器**，再安排原
  参数，因此它可能使后面原本恰好放得下的整个聚合改走栈；
- 所有 padding 写入时为零；读取时 padding 永不被当作值。

本模块是题面定义的合成 ABI，**不宣称**与 System V AMD64 或任何真实
平台 ABI 完整兼容（例如不实现 SSEUP、X87、按位域合并等规则）。

## 公开接口

类型构造：

| 名称 | 说明 |
|---|---|
| `IntType(size, signed)` | `size ∈ {1,2,4,8}`；常量 `INT8..INT64/UINT8..UINT64` |
| `FloatType()` / `DOUBLE` | 8 字节浮点 |
| `PtrType()` / `PTR` | 8 字节指针 |
| `ArrayType(elem, count)` | 显式长度同类数组 |
| `RecordType(name)` + `.define([(字段名, 类型), ...])` | 记录；支持先建壳后定义 |

参数编组：

| 函数 | 语义 |
|---|---|
| `layout(t) -> Layout` | `size/align` 及逐字段 `FieldLayout(name, offset, type, layout)` |
| `classify(t) -> tuple[str, ...]` | 逐 8 字节块 `INTEGER`/`FLOAT`；`()`（空）或 `(MEMORY,)` |
| `encode(t, value) -> bytes` / `decode(t, data) -> value` | 类型化小端字节编解码 |
| `pack_call([(Type, value), ...]) -> CallImage` | 编组为寄存器 + 栈映像 |
| `unpack_call(types, image) -> list` | 从映像重建参数值 |

返回编组：

| 函数 | 语义 |
|---|---|
| `return_convention(t, address=None) -> ReturnConvention` | 决定返回安放：`registers`/`memory`/`none` |
| `pack_invoke(return_type, args, memory=None, address=0, capacity=None) -> InvokeImage` | 调用方：编组一次带返回值的调用（间接返回时注入隐藏指针） |
| `prepare_return_buffer(t, memory, address, capacity)` | 验证缓冲区容量/对齐/范围，通过返回 `(memory, address)` |
| `encode_return(invoke, value) -> ReturnImage` | 被调方：按约定编码返回值（间接返回时一次性写入缓冲区） |
| `decode_return(invoke, ret) -> value` | 调用方：按约定读回返回值 |

`CallImage`：`int_regs`（4 槽，未用为 `None`，已用为该 8 字节块的小端
整数值）、`float_regs`（2 槽，double）、`stack`（16 字节对齐字节串）、
`placements`（逐参数的 `Placement(kind, eightbytes, int_regs, float_regs,
stack_offset, size)`，`kind ∈ {"registers", "stack", "none"}`）。

`InvokeImage`：`return_type`、`convention`（`ReturnConvention(kind,
eightbytes, size, align, address)`，`kind ∈ {"registers","memory","none"}`）、
`call_image`（间接返回时 `int_regs[0]` 已是隐藏指针）、`memory`/`address`
（间接返回的有界缓冲区，否则为 `None`）、`hidden_pointer`（隐藏指针值）。

`ReturnImage`：`kind`（`registers`/`memory`/`none`）、`eightbytes`、
`int_regs`（2 槽）、`float_regs`（2 槽），未用槽位为 `None`。

`Memory`：合成地址空间（`bytearray` 支撑，基地址 0）。`allocate(size,
align)` bump 分配并返回地址；`read/write/fill` 均做边界校验。间接返回的
缓冲区就分配在这里，地址即隐藏指针值。

值的 Python 表示：整数/指针 → `int`；浮点 → `float`；数组 → `list`；
记录 → `dict`（字段名 → 值，键集合必须与定义完全一致）。

## 错误语义

所有错误为 `ABIError` 子类，携带可定位的 `path`（如 `arg[2].samples[3]`）：

- `InvalidTypeError`：非法类型定义（未定义/重复定义记录、字段重名、
  尺寸超限等）；`RecursiveTypeError` 是其子类，专指循环记录类型；
- `InvalidValueError`：值与类型不匹配（含 **bool 冒充整数一律拒绝**、
  数组长度不符、记录字段缺失/多余）；
- `IntegerOverflowError`：整数/指针超出目标类型可表示范围；
- `NonFiniteFloatError`：浮点值为 NaN/Infinity；
- `BufferError`：间接返回缓冲区非法（容量不足、未对齐、地址范围越界）。

不做静默纠正：浮点字段只接受 `float`（拒绝 `int`，避免 `|v| ≥ 2**53`
时的静默精度损失）。`pack_call`/`pack_invoke` 任一参数非法即整体抛错，
不返回部分编组结果。间接返回时缓冲区的容量、对齐、范围在**任何写入
之前**完成验证；被调方 `encode_return` 还会复核缓冲区，返回值编码失败
（如越界整数）时缓冲区保持原样，**绝不写入部分结果**。

## 输入上限

| 上限 | 值 |
|---|---|
| 单次调用参数个数 `MAX_ARGS` | 1024 |
| 数组元素数 `MAX_ARRAY_COUNT` | 65536 |
| 记录字段数 `MAX_RECORD_FIELDS` | 256 |
| 任一类型布局总字节数 `MAX_TYPE_SIZE` | 1 MiB |
| 合成地址空间 `MAX_ADDRESS` | 2⁶⁴ − 1 |
| 整数返回寄存器 `RET_INT_REG_COUNT` | 2 |
| 浮点返回寄存器 `RET_FLOAT_REG_COUNT` | 2 |

超出即抛 `InvalidTypeError`/`InvalidValueError`。这些是接口边界，
demo 不穷举上限规模。

## 浮点误差口径

浮点以 IEEE-754 binary64 原样存取：任何有限 `float` 经
`encode → decode`（含经寄存器/栈映像的 `pack_call → unpack_call`）
**逐比特精确往返，误差为 0**，不使用任何容差。NaN/Infinity 在入口
即被拒绝，不进入映像。

## 设计取舍

- **整体回退而非逐块溢出**：寄存器不足时聚合整体走栈且不消耗寄存器，
  与题面契约一致，也使 `Placement` 语义简单可逆；
- **寄存器槽存原始 8 字节块**：`INTEGER` 块存小端整数、`FLOAT` 块存
  double，块内不足 8 字节零填充，因此混合类聚合可无损往返；
- **记录先建壳后 `define`**：使循环类型可表达，从而在 `layout` 时
  被 `RecursiveTypeError` 显式拒绝（直接、间接、经数组的循环均检测）；
- **返回寄存器独立成组**：整数/浮点返回寄存器各 2 个，与参数寄存器
  互不占用；不超过 16 字节的聚合至多两个块，返回寄存器必然够用；
- **隐藏指针即缓冲区地址**：间接返回时调用方缓冲区的地址直接作为隐藏
  指针编入第一个整数参数寄存器，被调方据此写入，无需额外元数据；
- **原子写入**：间接返回先把整个值编码成字节串（padding 全零），再用
  单次切片赋值写入缓冲区，中途失败不会留下部分结果；
- **无全局状态**：每次 `pack_call`/`pack_invoke` 独立计算，失败不留
  部分结果。

## 已知限制

- 仅实现题面子集：无位域、无联合、无可变长度数组；
- 标量参数在寄存器耗尽时同样走栈（题面只规定聚合的回退，此处对标量
  采用同一规则）；
- `unpack_call`/`decode_return` 信任映像内部一致性，仅校验栈长度、
  缓冲区范围与 kind 匹配等结构性错误；
- `Memory` 是合成地址空间（线性字节数组），仅用于演示间接返回的
  有界缓冲区语义，不模拟真实虚拟内存；
- 不对齐到任何真实硬件 ABI，不可直接用于真实 FFI 调用。

## 验证方式

- `tests/test_abi_marshal.py`（参数编组，42 用例）：期望值来自手工
  硬编码字面量与测试文件内**独立编写**的逐字段偏移参考
  `ref_record_layout`（不调用被测核心）；覆盖混合类聚合、整体回退、
  尾填充、嵌套数组、空记录约定、循环类型、整数溢出、bool/NaN 拒绝、
  错误路径定位；pack/unpack 往返一致仅作补充测试；
- `tests/test_return_abi.py`（返回编组，28 用例）：期望值来自手工
  硬编码的完整 16/32 字节布局字面量与独立偏移参考 `ref_offsets` /
  独立字节拼装 `ref_grid_bytes`（均不调用被测核心）；覆盖寄存器返回
  逐块核对、间接返回缓冲区逐字节核对、padding 写零且读取忽略、隐藏
  指针刚好挤掉最后一个聚合、缓冲区容量/对齐/范围拒绝且哨兵不变、
  返回值越界不写部分结果、完整“参数编码→模拟计算→返回编码→调用方
  读取”流程的独立字节核对；
- `demo.py`：真实计算并打印布局、参数/返回寄存器、栈映像、缓冲区字节，
  以及实际触发的越界拒绝（`u8 = 300`、返回值 `i16 = 32768`）、NaN 拒绝、
  缓冲区容量不足与未对齐拒绝。
