# abi_marshal — 聚合参数的合成 ABI 编组

离线外部函数接口工程的可复用模块：把 `(类型, 值)` 实参序列编组为
“寄存器 + 栈”的调用映像，并可从映像重建参数。

## 运行环境

- Python **3.14.7**，仅标准库，**无需安装任何额外依赖**；
- Linux 原生运行；核心、测试、演示全部离线，不需要网络、账号、密钥、
  数据库、中间件、云 API 或 Docker。

## 命令

```sh
python -m unittest discover -s tests -v   # 运行测试（65 个用例）
python demo.py                            # 固定输入演示（约 1 秒内完成）
python demo_iteration.py                  # 位域迭代演示（正常 + 边界，约 0.1 秒）
```

## venv 准备

仓库自带 `.venv`（Python 3.14.7）。如需重建：

```sh
python3.14 -m venv .venv        # 或：python3 -m venv .venv
.venv/bin/python -m unittest discover -s tests -v
```

无第三方依赖，`requirements.txt` 为空占位；venv 激活与否均可，
只要解释器是 Python 3.14.7。

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
- 记录可含**无符号位域**字段：底层存储单元仅 8/16/32 位无符号整数；
  小端、低位（LSB）先分配；字段不横跨单元，当前单元放不下即开新单元
  （按单元大小对齐）；零宽字段结束当前单元，下一字段重新对齐；未用位
  与结构体填充一律清零；位域字节按整数类参与 8 字节块分类。

本模块是题面定义的合成 ABI，**不宣称**与 System V AMD64 或任何真实
平台 ABI 完整兼容（例如不实现 SSEUP、X87 等规则；位域打包规则是
题内自定义规则，不对应任何真实编译器的位域 ABI）。

## 公开接口

类型构造：

| 名称 | 说明 |
|---|---|
| `IntType(size, signed)` | `size ∈ {1,2,4,8}`；常量 `INT8..INT64/UINT8..UINT64` |
| `FloatType()` / `DOUBLE` | 8 字节浮点 |
| `PtrType()` / `PTR` | 8 字节指针 |
| `ArrayType(elem, count)` | 显式长度同类数组 |
| `RecordType(name)` + `.define([(字段名, 类型), ...])` | 记录；支持先建壳后定义 |
| `BitfieldType(base, width)` | 无符号位域；`base` 为 `UINT8/16/32`，`width ∈ [0, base 位数]`；`width=0` 为零宽分隔字段 |

函数：

| 函数 | 语义 |
|---|---|
| `layout(t) -> Layout` | `size/align` 及逐字段 `FieldLayout(name, offset, type, layout, bit_offset, mask)`；位域字段给出字节位置、单元内位偏移与已就位掩码，非位域字段后两者为 `None` |
| `classify(t) -> tuple[str, ...]` | 逐 8 字节块 `INTEGER`/`FLOAT`；`()`（空）或 `(MEMORY,)` |
| `encode(t, value) -> bytes` / `decode(t, data) -> value` | 类型化小端字节编解码 |
| `pack_call([(Type, value), ...]) -> CallImage` | 编组为寄存器 + 栈映像 |
| `unpack_call(types, image) -> list` | 从映像重建参数值 |

`CallImage`：`int_regs`（4 槽，未用为 `None`，已用为该 8 字节块的小端
整数值）、`float_regs`（2 槽，double）、`stack`（16 字节对齐字节串）、
`placements`（逐参数的 `Placement(kind, eightbytes, int_regs, float_regs,
stack_offset, size)`，`kind ∈ {"registers", "stack", "none"}`）。

值的 Python 表示：整数/指针 → `int`；浮点 → `float`；数组 → `list`；
记录 → `dict`（字段名 → 值，键集合必须与定义完全一致；零宽位域
不持有值，不出现在字典中）。位域值为 `int`，范围 `[0, 2**width - 1]`，
超范围抛 `IntegerOverflowError`（`pack_call` 整次拒绝，不留部分结果）；
相邻位域写入互不影响（按掩码合并，未用位保持为零）。

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

## 已知限制

- 仅实现题面子集：无联合、无可变长度数组、无返回值编组；位域仅支持
  记录内的无符号 8/16/32 位单元，不支持 64 位单元、不支持位域数组；
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
- `tests/test_bitfields.py`：位域布局与编解码的期望字节由测试文件内
  **独立编写**的逐位组装/取出参考 `set_bits`/`get_bits` 生成（不调用
  被测核心）；覆盖跨单元分配、零宽分隔、嵌套记录、恰好 16 字节进
  寄存器与 20 字节走栈的分界、未用位/填充清零、相邻字段不互相覆盖、
  位域溢出整次拒绝，以及不含位域的旧类型布局保持不变；
- `demo_iteration.py`：现场真实计算位域记录的布局、编码字节、
  pack/unpack 重建、16/20 字节寄存器-栈分界，以及一次实际触发的
  位域越界拒绝（约 0.1 秒完成，无 sleep、无预录输出）；
- `demo.py`：真实计算并打印布局、寄存器、栈映像、参数重建，以及一次
  实际触发的越界拒绝（`u8 = 300`）与 NaN 拒绝。
