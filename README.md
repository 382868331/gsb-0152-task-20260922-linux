# abi_marshal — 聚合参数的合成 ABI 编组

为离线外部函数接口（FFI）工程提供的可复用 Python 库：按一套明示的合成 ABI，
把含聚合（记录/数组）的调用参数编组到寄存器与栈映像，并能从映像完整重建参数。

- **Python 版本**：3.14.7（仅标准库，零第三方依赖，安装 Python 即可运行）
- **平台**：Linux 原生；核心、测试、演示全部离线
- **包**：`abi_marshal/`（`model` 类型模型、`layout` 布局与分类、`marshal` 编组、`errors` 错误语义）

## 运行

```sh
python -m unittest discover -s tests -v   # 必要测试（含独立逐字段偏移参考）
python demo.py                            # 固定输入演示，约 0.1 秒内完成
```

## ABI 契约（固定，不宣称兼容任何行业标准 ABI）

- 小端；指针 8 字节（按无符号 64 位整数传输）；整数 1/2/4/8 字节（有/无符号）；浮点 8 字节（IEEE-754 binary64）。
- 记录自然对齐，对齐上限 8；记录尾部按对齐补填充；**空记录 size=0、align=1**，不消耗寄存器或栈。
- 数组为显式长度的同类元素序列，元素按完整 size 连续排列（stride == 元素 size）。
- 聚合按 8 字节块（eightbyte）分类：块内有整数/指针叶子则该块为整数类，否则浮点类；**超过 16 字节整体走栈**（MEMORY）。
- 整数寄存器 4 个、浮点寄存器 2 个，各 8 字节。标量各占 1 个对应寄存器。
- **一个聚合所需寄存器不足时，整体走 16 字节对齐的栈槽且不消耗寄存器**（后续参数仍可使用剩余寄存器）；溢出的标量走 8 字节对齐栈槽。
- 寄存器与栈承载原始字节，填充为零。

## 接口

```python
import abi_marshal as abi

# 类型
t = abi.RecordType("Mixed", [("tag", abi.I8), ("weight", abi.F64)])
arr = abi.ArrayType(abi.I16, 3)          # 显式长度
# 标量单例：I8 I16 I32 I64 U8 U16 U32 U64 F64 PTR

# 布局与分类
lay = abi.layout(t)        # Layout(size=16, align=8, fields=(FieldLayout(name, offset, type), ...))
cls = abi.classify(t)      # ('integer', 'float')；>16 字节返回 abi.MEMORY；size 0 返回 ()

# 编组 / 解组（值：标量用 int/float，记录用 dict，数组用 list/tuple）
img = abi.pack_call([abi.I64, t], [7, {"tag": 1, "weight": 2.5}])
# img.int_regs: 4 项（None 或 8 字节）；img.float_regs: 2 项；img.stack: bytes
# img.placements: 每个参数的落点（("int", i)/("float", i)/("stack", off) 逐 8 字节块）
vals = abi.unpack_call([abi.I64, t], img)   # 重建参数，与输入相等

# 单值编解码与调用计划（底层）
abi.encode_value(t, {"tag": 1, "weight": 2.5})
abi.decode_value(t, buf, 0)
abi.plan_call([abi.I64, t])                 # CallPlan(placements, stack_size)
```

## 错误语义（可定位，不静默纠正）

所有错误继承 `abi.ABIError`，携带 `path` 元组（如 `("arg0", 1, "x")`，渲染为 `arg0[1].x`）：

| 错误 | 触发 |
|---|---|
| `CycleError` | 记录按值（传递地）包含自身 |
| `IntegerOverflowError` | 整数/指针越界；int→f64 不能精确表示 |
| `NonFiniteFloatError` | 浮点参数为 NaN 或 ±Infinity |
| `ValueTypeError` | Python 值类型不符（**整数/指针/浮点均拒绝 bool**；记录缺字段或多字段） |
| `ArityError` | 参数个数或数组长度不符 |
| `TypeDefinitionError` | 类型对象非法（整数尺寸 ∉ {1,2,4,8}、数组长度越界、重复字段等） |
| `SizeLimitError` | 类型尺寸超过上限 |
| `ImageError` | `unpack_call` 的映像畸形（寄存器数/宽度不符、栈长度不符、所需寄存器为空） |

`pack_call` 是纯函数：任一参数非法即抛错，不产生部分映像、不留部分变更。

## 输入上限（接口边界，非目标规模）

- 数组长度：`0 ≤ count ≤ 2**20`（`MAX_ARRAY_COUNT`）
- 任意类型布局尺寸：`≤ 2**24` 字节（`MAX_TYPE_SIZE`，16 MiB）
- 寄存器聚合：`≤ 16` 字节；整数寄存器 4 个、浮点寄存器 2 个

## 浮点误差口径

本库不做任何浮点运算，只做二进制级编解码：`struct.pack/unpack('<d', v)` 逐位往返，
**误差恒为 0**（测试含 `0.1`、`-0.0`、次正规数 `2**-1074` 的逐位断言）。int 作为浮点参数时
仅在可精确表示（`|v| ≤ 2**53` 且转换可逆）时接受，否则抛 `IntegerOverflowError`，绝不静默舍入；
NaN/±Infinity 一律拒绝。

## 设计取舍

- 寄存器与栈一律承载**原始字节**而非解释值：块分类只决定寄存器种类，位模式不变，往返天然一致。
- 布局对类型对象做按次调用的记忆化；循环检测基于递归栈上的对象身份，共享子记录不算循环。
- 栈槽长度向上取整到 8 字节，聚合槽 16 字节对齐、标量槽 8 字节对齐；填充为零。
- `unpack_call` 仅凭类型重算落点，要求映像与计划**精确**匹配（栈长度相等、寄存器宽度 8），
  以尽早暴露调用方错误。
- 测试中的期望值来自测试文件内独立编写的逐字段偏移参考与手算常量，不调用被测核心；
  往返一致性仅作补充手段。

## 已知限制

- 仅实现本题契约的子集：无 f32/f16、位域、联合、可变参数、返回值编组、按引用聚合。
- 指针是不透明的 64 位无符号值，不解引用、不校验目标。
- 类型对象按身份（identity）比较；语义相同的两个 `RecordType` 不视为同一类型。
- 不宣称兼容 System V AMD64 或任何真实平台 ABI（分类规则只是与其神似的合成规则）。
