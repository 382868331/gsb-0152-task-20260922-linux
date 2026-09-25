# abi_marshal — 聚合参数与返回值的合成 ABI 编组

离线外部函数接口工程的可复用模块：把 `(类型, 值)` 实参序列编组为
“寄存器 + 栈”的调用映像，并可从映像重建参数；返回值沿用同一套块
分类，经返回寄存器或调用方提供的有界缓冲区传递。

## 运行环境

- Python **3.14.7**，仅标准库，**无需安装任何额外依赖**；
- Linux 原生运行；核心、测试、演示全部离线，不需要网络、账号、密钥、
  数据库、中间件、云 API 或 Docker。

## 命令

```sh
./.venv/bin/python -m unittest discover -s tests -v   # 运行测试（75 个用例）
./.venv/bin/python demo.py                            # 固定输入演示（约 1 秒内完成）
```

（任何 `python` 解释器 ≥ 3.14 均可；仓库内命令统一走 `./.venv/bin/python`。）

## 明示 ABI 契约（本模块实现且仅实现此子集）

参数传递：

- 小端字节序；指针 8 字节（无符号）；
- 整数 1/2/4/8 字节（有/无符号）；浮点 8 字节（IEEE-754 binary64）；
- 记录按自然对齐布局，对齐上限 8；空记录 `size=0, align=1`；
- 数组为显式长度的同类元素序列，元素按完整 size 连续排列；
- 聚合按 8 字节块分类：块内含任意整数/指针字节 → `INTEGER`，否则
  `FLOAT`；聚合超过 16 字节 → `MEMORY`（整体走栈）；
- 整数寄存器 4 个、浮点寄存器 2 个；聚合所需寄存器不足时**整体**走
  16 字节对齐的栈，且**不消耗任何寄存器**（不允许拆半安放）；
- 空记录不消耗寄存器也不消耗栈。

返回值传递：

- 标量及不超过 16 字节的聚合沿用同一套块分类，经返回寄存器传递：
  **整数类返回寄存器 2 个、浮点类返回寄存器 2 个**（至多 2 块，必然
  放得下）；
- 空记录（`size=0`）不占用任何返回寄存器；
- 超过 16 字节的聚合**间接返回**：调用方传入有界返回缓冲区
  `ReturnBuffer(address, size)`，被调方把隐藏指针（缓冲区地址）**先占
  整数参数寄存器 0**，再安排原参数——因此它可能把后面的整个聚合挤到
  栈上；
- 缓冲区的大小（≥ 返回类型 size、≤ 1 MiB）、对齐（地址按 16 对齐）、
  范围（`address + size` 不溢出 64 位地址空间）在**编码任何参数之前**
  先验证，失败不写入任何部分结果；返回类型不需要缓冲区时传入缓冲区
  同样报错；
- 所有 padding 写零；读取时 padding 不作为值解释（只按字段布局读取
  有效字节）。

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
| `pack_call([(Type, value), ...], ret_type=None, ret_buffer=None) -> CallImage` | 编组为寄存器 + 栈映像；指定 `ret_type` 时同时规划返回值 |
| `unpack_call(types, image) -> list` | 从映像重建参数值 |

返回值编组：

| 函数 / 类型 | 语义 |
|---|---|
| `plan_return(ret_type) -> ReturnPlan` | 返回安放方案：`kind ∈ {"registers", "indirect", "none"}` |
| `ReturnBuffer(address, size)` | 有界返回缓冲区（地址 16 对齐，容量 ≤ 1 MiB） |
| `pack_return(ret_type, value, plan=None) -> ReturnImage` | 返回值编组为返回寄存器映像或待写缓冲区的字节串 |
| `unpack_return(ret_type, image, plan=None) -> value` | 调用方从返回映像读取返回值 |
| `simulate_call(func, args, ret_type, ret_buffer=None) -> value` | 完整流程：参数编组 → 解参与计算 → 返回编组 → 调用方读取 |

`CallImage`：`int_regs`（4 槽，未用为 `None`，已用为该 8 字节块的小端
整数值）、`float_regs`（2 槽，double）、`stack`（16 字节对齐字节串）、
`placements`（逐参数的 `Placement(kind, eightbytes, int_regs, float_regs,
stack_offset, size)`，`kind ∈ {"registers", "stack", "none"}`）、
`ret_plan`（返回方案，未指定返回类型时为 `None`）、`hidden_ptr`（间接
返回时隐藏指针的安放结果，先于所有用户参数占整数寄存器 0，**不计入**
`placements`）。

`ReturnImage`：`plan`、`int_regs`（2 槽）、`float_regs`（2 槽）、
`memory`（间接返回时写入缓冲区地址处的字节串，其余为 `b""`）。

值的 Python 表示：整数/指针 → `int`；浮点 → `float`；数组 → `list`；
记录 → `dict`（字段名 → 值，键集合必须与定义完全一致）。

## 错误语义

所有错误为 `ABIError` 子类，携带可定位的 `path`（如 `arg[2].samples[3]`、
`return`、`ret_buffer`）：

- `InvalidTypeError`：非法类型定义（未定义/重复定义记录、字段重名、
  尺寸超限等）；`RecursiveTypeError` 是其子类，专指循环记录类型；
- `InvalidValueError`：值与类型不匹配（含 **bool 冒充整数一律拒绝**、
  数组长度不符、记录字段缺失/多余、缓冲区缺失/多余/未对齐/容量不足/
  地址范围溢出）；
- `IntegerOverflowError`：整数/指针超出目标类型可表示范围（参数与
  返回值同样校验）；
- `NonFiniteFloatError`：浮点值为 NaN/Infinity。

不做静默纠正：浮点字段只接受 `float`（拒绝 `int`，避免 `|v| ≥ 2**53`
时的静默精度损失）。`pack_call`/`pack_return` 任一输入非法即整体抛错，
不返回部分编组结果（无副作用状态，不存在部分变更）；缓冲区验证发生
在任何参数编码之前。

## 输入上限

| 上限 | 值 |
|---|---|
| 单次调用参数个数 `MAX_ARGS` | 1024 |
| 数组元素数 `MAX_ARRAY_COUNT` | 65536 |
| 记录字段数 `MAX_RECORD_FIELDS` | 256 |
| 任一类型布局总字节数 `MAX_TYPE_SIZE` | 1 MiB |
| 返回缓冲区容量 `MAX_RET_BUFFER_SIZE` | 1 MiB |

超出即抛 `InvalidTypeError`/`InvalidValueError`。这些是接口边界，
demo 不穷举上限规模。

## 浮点误差口径

浮点以 IEEE-754 binary64 原样存取：任何有限 `float` 经
`encode → decode`（含经寄存器/栈映像的 `pack_call → unpack_call`、
经返回寄存器/缓冲区的 `pack_return → unpack_return`）
**逐比特精确往返，误差为 0**，不使用任何容差。NaN/Infinity 在入口
即被拒绝，不进入映像。

## 设计取舍

- **整体回退而非逐块溢出**：寄存器不足时聚合整体走栈且不消耗寄存器，
  与题面契约一致，也使 `Placement` 语义简单可逆；
- **寄存器槽存原始 8 字节块**：`INTEGER` 块存小端整数、`FLOAT` 块存
  double，块内不足 8 字节零填充，因此混合类聚合可无损往返；
- **返回寄存器独立成组**：2 整数 + 2 浮点，与参数寄存器互不干扰；
  ≤16 字节的返回聚合至多 2 块，必然放得下，无需返回值回退规则；
- **隐藏指针先于用户参数**：间接返回的缓冲区地址占整数参数寄存器 0，
  用户参数顺延，可能整体挤到栈上；隐藏指针记录在
  `CallImage.hidden_ptr`，不计入 `placements`，`unpack_call` 的用户
  参数视图不受影响；
- **缓冲区先验证后编码**：大小、对齐、范围在编码任何参数之前检查，
  失败时调用映像根本不存在，谈不上部分写入；
- **记录先建壳后 `define`**：使循环类型可表达，从而在 `layout` 时
  被 `RecursiveTypeError` 显式拒绝（直接、间接、经数组的循环均检测）；
- **无全局状态**：每次 `pack_call`/`pack_return` 独立计算，失败不留
  部分结果。

## 已知限制

- 仅实现题面子集：无位域、无联合、无可变长度数组；
- 标量参数在寄存器耗尽时同样走栈（题面只规定聚合的回退，此处对标量
  采用同一规则）；
- `unpack_call`/`unpack_return` 信任映像内部一致性，仅校验栈长度、
  寄存器空槽、plan 匹配等结构性错误；
- 间接返回的“写入缓冲区”以 `ReturnImage.memory` 字节串模拟，不操作
  真实内存；
- 不对齐到任何真实硬件 ABI，不可直接用于真实 FFI 调用。

## 验证方式

- `tests/test_abi_marshal.py`（42 个用例，参数编组回归）：期望值来自
  手工硬编码字面量与测试文件内**独立编写**的逐字段偏移参考
  `ref_record_layout`（不调用被测核心）；覆盖混合类聚合、整体回退、
  尾填充、嵌套数组、空记录约定、循环类型、整数溢出、bool/NaN 拒绝、
  错误路径定位；
- `tests/test_abi_return.py`（33 个用例，返回值编组）：期望值同样为
  手工硬编码字节串/寄存器值，或由测试**手工构造** `ReturnImage` 后
  单向读取；覆盖混合整浮记录返回、嵌套数组间接返回的字节级布局、
  padding 写零与读取忽略、隐藏指针挤占最后一个整数寄存器、缓冲区
  大小/对齐/范围验证及其先于参数编码的顺序、返回值整数越界、空记录
  返回不占寄存器；pack/unpack 往返一致仅作补充测试；
- `demo.py`：真实计算并打印布局、寄存器、栈映像、参数重建、完整
  调用流程（参数编码 → 模拟计算 → 返回编码 → 调用方读取），以及
  实际触发的越界拒绝（`u8 = 300`）、NaN 拒绝、返回值越界拒绝与
  缓冲区未对齐拒绝。
