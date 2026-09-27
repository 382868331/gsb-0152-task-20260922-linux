# abi_marshal — 聚合参数的合成 ABI 编组

离线外部函数接口工程的可复用模块：把 `(类型, 值)` 实参序列编组为
“寄存器 + 栈”的调用映像，并可从映像重建参数。

## 运行环境

- Python **3.14.7**，仅标准库，**无需安装任何额外依赖**；
- Linux 原生运行；核心、测试、演示全部离线，不需要网络、账号、密钥、
  数据库、中间件、云 API 或 Docker。

## 命令

```sh
python -m unittest discover -s tests -v   # 运行测试（58 个用例）
python demo.py                            # 固定输入演示（约 1 秒内完成）
```

## 明示 ABI 契约（本模块实现且仅实现此子集）

- 小端字节序；指针 8 字节（无符号）；
- 整数 1/2/4/8 字节（有/无符号）；浮点 8 字节（IEEE-754 binary64）；
- 记录按自然对齐布局，对齐上限 8；空记录 `size=0, align=1`；
- 数组为显式长度的同类元素序列，元素按完整 size 连续排列；
- 聚合按 8 字节块分类：块内含任意整数/指针字节 → `INTEGER`，否则
  `FLOAT`；聚合超过 16 字节 → `MEMORY`（整体走栈）；
- 整数寄存器 4 个、浮点寄存器 2 个；聚合所需寄存器不足时**整体**走
  16 字节对齐的栈，且**不消耗任何寄存器**（不允许拆半安放）；
- 空记录不消耗寄存器也不消耗栈。

返回值沿用完全相同的分类、对齐与填充规则：

- 非空且 <=16 字节的标量/聚合为**小结果**：逐 8 字节块从 0 号结果
  寄存器起安放（INTEGER 块进整数寄存器、FLOAT 块进浮点寄存器，
  末块不足 8 字节零填充）；
- >16 字节（MEMORY）为**大结果**：由**调用方**提供一块按该类型自然
  对齐的缓冲区，其 8 字节地址作为**隐式指针实参占用第一个参数位置**，
  其后的所有显式实参顺延一个寄存器/栈位置（寄存器耗尽时同样走栈）；
  缓冲区在合成模型中就是 `CallImage.result_buffer` 字节串，地址是
  合成常量 `RESULT_BUFFER_BASE = 0x7F00_0000_0000`，不对应真实内存；
- 空记录没有返回值，不占寄存器也不占缓冲区，显式实参位置不变。

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

函数：

| 函数 | 语义 |
|---|---|
| `layout(t) -> Layout` | `size/align` 及逐字段 `FieldLayout(name, offset, type, layout)` |
| `classify(t) -> tuple[str, ...]` | 逐 8 字节块 `INTEGER`/`FLOAT`；`()`（空）或 `(MEMORY,)` |
| `encode(t, value) -> bytes` / `decode(t, data) -> value` | 类型化小端字节编解码 |
| `pack_call([(Type, value), ...]) -> CallImage` | 编组为寄存器 + 栈映像 |
| `unpack_call(types, image) -> list` | 从映像重建参数值 |
| `pack_callsite(return_type, args) -> CallImage` | 调用点编组：额外安排返回值接收位置，大结果时预置隐式 sret 首参 |
| `write_result(return_type, image, value) -> CallImage` | 被调方侧：把返回值按规则写入结果寄存器或调用方缓冲区 |
| `unpack_result(return_type, image) -> value` | 调用方侧：按同一套布局/对齐规则重建返回值 |

`CallImage`：`int_regs`（4 槽，未用为 `None`，已用为该 8 字节块的小端
整数值）、`float_regs`（2 槽，double）、`stack`（16 字节对齐字节串）、
`placements`（逐参数的 `Placement(kind, eightbytes, int_regs, float_regs,
stack_offset, size, sret)`，`kind ∈ {"registers", "stack", "none"}`，
`sret=True` 标记大结果缓冲区地址的隐式首参）、`result_placement`
（`pack_callsite` 产物中的 `ResultPlacement(kind, eightbytes, int_regs,
float_regs, address, size, align)`；`pack_call` 产物为 `None`）、
`result_buffer`（大结果缓冲区，初值全零）、`arg_base`（隐式实参个数，
0 或 1；`unpack_call` 会自动跳过前 `arg_base` 个槽）。

值的 Python 表示：整数/指针 → `int`；浮点 → `float`；数组 → `list`；
记录 → `dict`（字段名 → 值，键集合必须与定义完全一致）。

## 错误语义

所有错误为 `ABIError` 子类，携带可定位的 `path`（如 `arg[2].samples[3]`）：

- `InvalidTypeError`：非法类型定义（未定义/重复定义记录、字段重名、
  尺寸超限等）；`RecursiveTypeError` 是其子类，专指循环记录类型；
- `InvalidValueError`：值与类型不匹配（含 **bool 冒充整数一律拒绝**、
  数组长度不符、记录字段缺失/多余）；
- `IntegerOverflowError`：整数/指针超出目标类型可表示范围；
- `NonFiniteFloatError`：浮点值为 NaN/Infinity。

不做静默纠正：浮点字段只接受 `float`（拒绝 `int`，避免 `|v| ≥ 2**53`
时的静默精度损失）。`pack_call` 任一参数非法即整体抛错，不返回部分
编组结果（无副作用状态，不存在部分变更）。

## 输入上限

| 上限 | 值 |
|---|---|
| 单次调用参数个数 `MAX_ARGS` | 1024 |
| 数组元素数 `MAX_ARRAY_COUNT` | 65536 |
| 记录字段数 `MAX_RECORD_FIELDS` | 256 |
| 任一类型布局总字节数 `MAX_TYPE_SIZE` | 1 MiB |

超出即抛 `InvalidTypeError`/`InvalidValueError`。这些是接口边界，
demo 不穷举上限规模。

## 浮点误差口径

浮点以 IEEE-754 binary64 原样存取：任何有限 `float` 经
`encode → decode`（含经寄存器/栈映像的 `pack_call → unpack_call`，
以及经结果寄存器/调用方缓冲区的
`write_result → unpack_result`）**逐比特精确往返，误差为 0**，不使用
任何容差。NaN/Infinity 在入口即被拒绝，不进入映像或结果缓冲区。

## 设计取舍

- **整体回退而非逐块溢出**：寄存器不足时聚合整体走栈且不消耗寄存器，
  与题面契约一致，也使 `Placement` 语义简单可逆；
- **寄存器槽存原始 8 字节块**：`INTEGER` 块存小端整数、`FLOAT` 块存
  double，块内不足 8 字节零填充，因此混合类聚合可无损往返；
- **记录先建壳后 `define`**：使循环类型可表达，从而在 `layout` 时
  被 `RecursiveTypeError` 显式拒绝（直接、间接、经数组的循环均检测）；
- **无全局状态**：每次 `pack_call`/`pack_callsite` 独立计算，
  `write_result` 返回新映像而非原地改写，失败不留部分结果；
- **大结果用合成缓冲区表示**：调用方“提供空间”在模型中具现为
  `CallImage.result_buffer` 与固定合成地址 `RESULT_BUFFER_BASE`，
  其指针作为隐式首参参与与普通实参完全相同的寄存器/栈分配，因此
  sret 占满寄存器导致后续实参走栈等移位效应自然产生，无需特判。

## 已知限制

- 仅实现题面子集：无位域、无联合、无可变长度数组；
- 标量参数在寄存器耗尽时同样走栈（题面只规定聚合的回退，此处对标量
  采用同一规则）；
- 返回值是合成调用约定：不模拟真实 ABI 的 RAX 回传 sret 地址、
  调用方保存寄存器等细节；`write_result` 只覆写结果寄存器，其他
  寄存器槽保留进入时的值，故应先 `unpack_call` 取参、再写返回值；
- 结果缓冲区按返回类型**自然对齐**（对齐上限 8）合成，不做 16 字节
  堆对齐（合成模型中无真实分配器）；
- `unpack_call` 信任 `CallImage` 内部一致性，仅校验栈长度等结构性错误；
- 不对齐到任何真实硬件 ABI，不可直接用于真实 FFI 调用。

## 验证方式

- `tests/test_abi_marshal.py`：期望值来自手工硬编码字面量与测试文件内
  **独立编写**的逐字段偏移参考 `ref_record_layout`（不调用被测核心）；
  覆盖混合类聚合、整体回退、尾填充、嵌套数组、空记录约定、循环类型、
  整数溢出、bool/NaN 拒绝、错误路径定位；返回值部分覆盖小结果分块
  入寄存器、16 字节边界、大结果 sret 隐参移位（含寄存器耗尽走栈）、
  嵌套记录/数组结果的对齐填充字节独立校验、类型不匹配与未写结果
  寄存器拒绝；pack/unpack 往返一致仅作补充测试；
- `demo.py`：真实计算并打印布局、寄存器、栈映像、参数重建、小/大/
  空三类返回值的写入与重建（大结果展示隐式 sret 首参导致的移位），
  以及一次实际触发的越界拒绝（`u8 = 300`）与 NaN 拒绝。
