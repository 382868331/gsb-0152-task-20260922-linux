# abi_marshal — 聚合参数的合成 ABI 编组

离线外部函数接口工程的可复用模块：把 `(类型, 值)` 实参序列编组为
“寄存器 + 栈”的调用映像，并可从映像重建参数。

## 运行环境

- Python **3.14.7**，仅标准库，**无需安装任何额外依赖**；
- Linux 原生运行；核心、测试、演示全部离线，不需要网络、账号、密钥、
  数据库、中间件、云 API 或 Docker。

## 命令

```sh
python -m unittest discover -s tests -v   # 运行测试（49 个用例）
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

`pack_call` 另接受可选的 `return_type=` 与 `result_buffer=`；返回类型超过
16 字节时，调用方提供结果空间，其地址作为隐藏的第一个 `ptr` 参数。

返回值接口：

| 名称 | 说明 |
|---|---|
| `ResultBuffer(t, address=DEFAULT_RESULT_ADDRESS)` | 调用方为大结果提供的合成空间；地址必须 16 字节对齐，容量等于完整布局大小（含嵌套记录/数组间隙和尾填充） |
| `pack_return(t, value, result_buffer=None) -> ReturnImage` | 小结果按 8 字节块进寄存器；大结果写入结果空间并在整数寄存器 0 回传地址；空记录不占资源 |
| `unpack_return(t, image, result_buffer=None)` | 按同一布局、对齐与填充规则重建返回值 |

`ReturnImage.kind` 为 `registers` / `memory` / `none`；`int_regs` 与
`float_regs` 长度同样固定为 4 和 2。大结果场景的 `CallImage.sret_pointer`
是隐藏地址自身的安放记录，`CallImage.placements` 仍只列显式参数。

`CallImage`：`int_regs`（4 槽，未用为 `None`，已用为该 8 字节块的小端
整数值）、`float_regs`（2 槽，double）、`stack`（16 字节对齐字节串）、
`placements`（逐参数的 `Placement(kind, eightbytes, int_regs, float_regs,
stack_offset, size)`，`kind ∈ {"registers", "stack", "none"}`）。

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
- **无全局状态**：每次 `pack_call` 独立计算，失败不留部分结果。
- **大结果空间是合成字节缓冲**：`ResultBuffer` 只模拟“调用方提供地址、被
  调方写入完整布局”的协议，不读写真实地址；
- **返回值与参数共用同一套分类和重建规则**：嵌套记录、数组、自然对齐、
  间隙和尾填充都由 `encode/decode` 统一处理；
- **隐藏地址先占参数位置**：大返回值的 `ptr` 地址先于显式参数申请整数
  寄存器/栈，因此显式参数按原始 `pack_call` 规则整体顺延。

## 已知限制

- 仅实现题面子集：无位域、无联合、无可变长度数组；
- 标量参数在寄存器耗尽时同样走栈（题面只规定聚合的回退，此处对标量
  采用同一规则）；
- `unpack_call` 信任 `CallImage` 内部一致性，仅校验栈长度等结构性错误；
- 不对齐到任何真实硬件 ABI，不可直接用于真实 FFI 调用。

## 验证方式

- `tests/test_abi_marshal.py`：期望值来自手工硬编码字面量与测试文件内
  **独立编写**的逐字段偏移参考 `ref_record_layout`（不调用被测核心）；
  覆盖混合类聚合、整体回退、尾填充、嵌套数组、空记录约定、循环类型、
  整数溢出、bool/NaN 拒绝、错误路径定位；pack/unpack 往返一致仅作
  补充测试；
- `demo.py`：真实计算并打印布局、寄存器、栈映像、参数重建、小返回值/大
  返回值编组（含嵌套记录数组的对齐填充），以及一次实际触发的越界拒绝
  （`u8 = 300`）与 NaN 拒绝。
