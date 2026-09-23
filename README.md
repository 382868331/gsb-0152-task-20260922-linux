# abi_marshal — 聚合参数的合成 ABI 编组（v1 + v2 双布局）

离线外部函数接口工程的可复用模块：把 `(类型, 值)` 实参序列编组为
“寄存器 + 栈”的调用映像，并可从映像重建参数、在两种目标布局之间做
A→B 调用映像转换。

本仓库在原有 v1（小端单一合成 ABI）基础上**增量迭代出 v2**：显式描述
两种相反字节序的目标布局，新增带标签联合、跨存储单元位域、可变参数
默认提升、返回值间接区、变参寄存器保存区、按位共享存储单元、位级布局
证据与独立证书、A↔B 桥接。**v1 的接口、行为、异常类型与确定性结果全部
保留，旧调用零改动**；v2 能力通过显式描述符 `ABIDescriptor` 与独立入口
使用，不改变任何 v1 默认行为。

## 运行环境

- Python **3.14.7**，仅标准库，**无需安装任何额外依赖**；
- Linux 原生运行；核心、测试、演示全部离线，不读账号/密钥/配置备份/
  环境变量清单，不需要网络、数据库、容器、硬件或第三方 API；不调用
  本机 FFI（不使用 `ctypes`/`struct.Struct.native` 等主机布局）。

## 命令与预算

| 用途 | 命令 | 规模/耗时（本机实测） |
|---|---|---|
| **快速回归**（v1 42 + v2 快速 36 = 78 用例） | `python -m unittest tests.test_abi_marshal tests.test_abi_v2 -v` | ≈ 0.03 s |
| 全部测试（再含独立参考扩展验证 9 用例，共 **87**） | `python -m unittest discover -s tests -v` | ≈ 1.2 s |
| v1 固定输入演示 | `python demo.py` | ≈ 0.05 s |
| **v2 新能力演示**（含正常结果、棘手边界、一次真实拒绝） | `python demo_v2.py` | ≈ 0.06 s |

两套测试有意分开：`tests/test_abi_marshal.py` 与 `tests/test_abi_v2.py`
是确定性快速回归；`tests/test_abi_v2_extended.py` 是依赖独立逐位参考的
差分/穷举/主规模扩展验证。`discover` 会跑到全部，但快速回归可单独执行。

---

# 第一部分：v1（原契约，行为不变）

## 明示 ABI 契约（v1 实现且仅实现此子集）

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
平台 ABI 完整兼容。

## v1 公开接口

类型构造：`IntType(size, signed)`（常量 `INT8..INT64/UINT8..UINT64`）、
`FloatType()`/`DOUBLE`、`PtrType()`/`PTR`、`ArrayType(elem, count)`、
`RecordType(name).define([...])`。

函数：`layout(t)`、`classify(t)`、`encode(t, value)`、`decode(t, data)`、
`pack_call([(Type, value), ...]) -> CallImage`、
`unpack_call(types, image)`。

`CallImage`：`int_regs`（4 槽）、`float_regs`（2 槽）、`stack`
（16 字节对齐）、`placements`（逐参数 `Placement`）。

值表示：整数/指针 → `int`；浮点 → `float`；数组 → `list`；记录 → `dict`。

## v1 错误语义

均为 `ABIError` 子类，携带可定位 `path`（如 `arg[2].samples[3]`）：
`InvalidTypeError`（含子类 `RecursiveTypeError`）、`InvalidValueError`
（含子类 `IntegerOverflowError`、`NonFiniteFloatError`）。整数拒绝
`bool`；v1 入口拒绝 NaN/Infinity；浮点字段只接受 `float`。
任一参数非法即整体抛错，无部分编组。

## v1 输入上限

| 上限 | 值 |
|---|---|
| `MAX_ARGS` | 1024 |
| `MAX_ARRAY_COUNT` | 65536 |
| `MAX_RECORD_FIELDS` | 256 |
| `MAX_TYPE_SIZE` | 1 MiB |

## v1 浮点误差口径

有限 `float` 经 `encode → decode`（含 `pack_call → unpack_call`）
**逐比特精确往返，误差为 0**，无容差。NaN/Infinity 在入口即被拒绝。

---

# 第二部分：v2（显式双布局 / 联合 / 位域 / 变参 / 桥接 / 证书）

## 1. 目标布局由描述符显式声明（不靠主机 struct 推导）

`ABIDescriptor(name, endian, max_align, int_regs, stack_align, pointer_size,
save_int_slots, save_float_slots)`，其中 `int_regs` 是
`RegisterRules(int_count, float_count, max_aggregate_bytes)`。内置：

| 描述符 | 字节序 | 聚合对齐上限 | 整数/浮点寄存器 | 寄存器聚合界 | 变参保存区（int/float 槽） |
|---|---|---|---|---|---|
| `LAYOUT_A` | little | 8 | 4 / 2 | 16 B | 4 / 2 |
| `LAYOUT_B` | **big** | **16** | **6 / 4** | **32 B** | 6 / 4 |

两者字节序相反、对齐上限与寄存器规则都不同。布局的一切几何量都来自
描述符；`desc.flipped()` 只翻转字节序（证书重放用）。用户也可用同一
构造器声明其它布局，但 v2 从不读取主机原生对齐/端序。

## 2. 类型与值的新增构造

| 名称 | 说明 |
|---|---|
| `Int128Type(signed)` / `INT128` / `UINT128` | 16 字节整数 |
| `Float32Type()` / `FLOAT32` | 4 字节 IEEE-754 binary32（**仅 v2**；v1 不支持） |
| `BitField(name, base, width, lsb=None)` | 记录内位域。`base` 为 1/2/4/8 字节 `IntType`，决定存储单元；`width` 1..单元位数；`lsb` 省略时顺序打包 |
| `UnionType(name, tag_type=UINT8).define([(名, 类型), ...])` | 带活动成员标签的联合；`tag_type=None` 为无标签联合 |
| `UnionValue(tag, value)` | 联合值；`tag` 为成员下标（int）或成员名（str） |
| `FloatBits(pattern, width=64)` | NaN 的**原始位载荷**（f64/f32），用于逐位保留 NaN payload；仅接受 NaN 位型 |
| `Param(type, variadic=False, shared=None, shared_size=None, tag_hint=None)` | 实参规约 |
| `SharedUnit(unit_id, size, views)` | 按位共享存储单元运行时值；`views={类型名: 值}` |

记录字段中放位域：`r.define([("x", BitField("x", UINT16, 5)), ...])`
（元组名必须与位域名一致）。

## 3. 布局规则（缺省选择已固定，结果确定）

- **标量对齐**：自然对齐；记录/数组的整体对齐 = min(成员最大对齐,
  描述符 `max_align`)；空记录 `size=0, align=1`。
- **位域**：连续且 `base` **宽度相同**的位域（signed 不影响分组）归入
  一个存储单元流；顺序打包时若当前单元剩余位不足，该位域**自动跨入
  下一个同型存储单元**（一个位域可产生两段 `BitSeg`）。给了显式 `lsb`
  时从单元绝对位起放；与已占位重叠 → `OverlappingBitFieldError`。
  位号相对“存储单元整数”的 LSB，与字节序无关；单元整数本身按描述符
  字节序存放。有符号性由 `base.signed` 决定，解码时按位宽符号扩展。
- **联合**：默认偏移 0 放 1 字节无符号标签，成员体从标签之后对齐起放，
  所有成员重叠；未激活成员字节写 0。无标签联合解码时若活动成员不能
  唯一辨识 → `InvalidUnionTagError`，可用 `tag_hint` 指定。
- **聚合分类**：逐 8 字节块，块内含任意整数/指针位即 INTEGER，否则
  FLOAT，纯填充块按 INTEGER；**联合按所有成员的最坏情况分类**。
- **寄存器安放**：聚合尺寸 ≤ 寄存器聚合界才可能走寄存器；申请整数/
  浮点块数，任一不足则**整体走 16 字节对齐栈且不消耗任何寄存器**
  （沿用 v1 的整体回退，两端口规则不同但策略相同）。8 字节块在寄存器
  槽内**左对齐**，字节序在块内；不足 8 字节零填充。
- **空聚合**：不占寄存器、不占栈（`kind="none"`）。

## 4. 同一调用帧的三个区域

帧几何：`[返回值间接区] [实参栈槽] [变参寄存器保存区]`，由纯几何的
`plan_call(params, desc, ret_type) -> PlannedCall` 给出，**规划器不接触
任何值**：

- **返回值间接区**：返回类型超过寄存器聚合界，或为 128 位整数时，经
  调用者帧偏移 0 的间接区返回（`ReturnInfo.indirect=True, area_offset=0`）；
  小返回走返回寄存器。
- **变参寄存器保存区**：仅当存在 `variadic=True` 实参时分配；每个进入
  寄存器的变参 8 字节块都按其**寄存器槽号**复制到保存区
  （`save_int_slots*8` + `save_float_slots*8`，位于实参栈槽之后）。
- **按位共享存储单元**：同 `shared` id 的多个实参构成一个单元；**首个
  实参为属主**，只占一份资源（寄存器或栈），其余实参 `kind="alias"`、
  不占资源。单元分类取所有视图位段的并集。多个视图在重叠位上写入的
  值必须逐位一致，否则 `OverlappingStorageError`（定位到具体视图实参
  路径）。共享视图不可为变参。

## 5. 可变参数默认提升

`variadic=True` 实参按合成 C 规则在编组前提升：1/2 字节整数 → 4 字节
同符号整数；`f32 → f64`（先舍入到 binary32 再展宽，f32 NaN 经
`FloatBits.widen_to_64` 保留 sign/quiet/payload）；4/8 字节整数、
指针、f64 原样。**128 位整数、聚合、数组无法表示** →
`VariadicPromotionError`（定位到 `arg[i]`）。解组时还原回原类型的规范
值（小整数按原符号性截断/扩展，f32 还原为 binary32）。

## 6. 编组 / 解组 / 返回入口

| 函数 | 语义 |
|---|---|
| `layout_v2(t, desc) -> LayoutV2` | 显式布局 + 逐位静态 `BitSeg` 证据 |
| `classify_v2(t, desc)` | 描述符下逐 8 字节块分类（纯几何） |
| `encode_v2(t, value, desc)` / `decode_v2(t, data, desc, tag_hint=...)` | 类型化字节编解码（NaN 叶子还原为 `FloatBits`） |
| `plan_call(params, desc, ret_type=None)` | **纯几何**规划：安放、区域、保存区、帧尺寸 |
| `pack_call_v2(params, values, desc, ret_type=None, ret_value=None, shared_units=()) -> CallImageV2` | 三阶段编组（全部成功才产出） |
| `unpack_call_v2(params, image, desc)` | 重建规范实参值 |
| `unpack_return_v2(ret_type, image, desc)` | 取回返回值（间接区或返回寄存器） |

`pack_call_v2` 三阶段、失败不留部分结果：① 每个实参离线编码（共享视图
位置传 `None`）；② 共享单元逐位合并（冲突整体失败）；③ 一次性写入帧
与寄存器。`CallImageV2` 含 `int_regs`（None 或无符号 64 位位型）、
`float_regs`（None / float / `FloatBits`）、`frame`（返回区+栈槽+保存区）、
`evidence`（逐位 `EvidenceSeg`：映像位置 → 参数路径）、`return_regs`、
`return_area`、`plan`。

## 7. A→B 调用映像桥接（先规范值后编组）

`bridge_call(src_image, params, src_desc, dst_desc, ret_type=None,
shared_units=(), ret_value=None)` 与显式写缓冲版本
`bridge_call_into(dst_buffer, ...)`：

1. 从源映像**解码为规范值**（联合按标签取活动成员；NaN 还原为
   `FloatBits` 携带原始 payload）；
2. 按目标描述符**重新规划并编码**；
3. 一次性产出目标映像。

比较口径是**规范值与约定保留的位**（有符号边界、NaN 原始载荷位、
活动成员、共享视图值），**不把填充字节相等当标准**（填充位编码固定
写 0，从不参与相等判定）。

**失败原子性**：解码/标签/提升/范围/共享冲突在写入前全部完成；
`bridge_call_into` 先在临时空间完成全部工作，成功才拷贝，因此任何
`BridgeError` 都保证目标缓冲区**零改写**。`BridgeError.reason` 取值：
`tag-out-of-range`、`overlap-conflict`、`unrepresentable-promotion`、
`value-range` 等，错误带 `path` 参数路径。

## 8. 位级证据与独立证书（检查器不信任自报布尔标志）

- 每个数据位都在 `CallImageV2.evidence` 中有来源（`ireg`/`freg`/
  `frame` + 槽/字节 + 位 + 宽度 + 参数路径 + 角色）。大端下证据位按
  物理字节映射（标量字节在 8 字节寄存器词的高字节侧；位域仍按单元
  整数的 LSB 编号）。
- `certify_layout(t, desc)`：独立重算几何，核对位段在范围内、同冲突域
  内不重叠（联合不同成员属不同域，天然可重叠）、非联合类型叶子位被
  恰为覆盖。
- `certify_frame(params, plan, desc)`：核对帧区域两两不重叠且不越界、
  返回间接区在偏移 0 且尺寸/对齐正确、栈槽 16 字节对齐、保存槽与变参
  寄存器一一对应，并**用仅翻转字节序的描述符独立重放**，确认返回值/
  实参对齐约束在两种字节序下都满足。
- `verify_image(params, values, image, desc, shared_units, ret_value)`：
  **独立重编码每个实参**与映像逐字节比对、用证据位重建并核对每位实际
  值（共享同一位的多路径只要求值一致）、核对保存区整词、双字节序重放、
  返回值一致；被篡改的映像会被检出（证书不读取任何求解器自报标志）。

## 9. v2 错误类型

`OverlappingBitFieldError`（InvalidTypeError）、
`InvalidUnionTagError`、`VariadicPromotionError`、
`OverlappingStorageError`（均 InvalidValueError）、
`BridgeError(ABIError)`。全部带可定位 `path`。

---

# 兼容性、差异与迁移

- **v1 完全保留**：`layout/classify/encode/decode/pack_call/unpack_call`
  及 `CallImage/Placement`、42 个旧测试、旧异常类型与确定性结果均未改。
  旧代码无需任何改动。
- **新能力是独立版本/显式选项**：只有显式传 `ABIDescriptor` 的 v2
  入口才启用双布局/联合/位域/变参/128 位/f32/共享/桥接；v1 默认行为
  不受影响。
- **f32 仅 v2**：`FLOAT32` 是新类型，v1 的 `encode/pack_call` 不接受它
  （v1 契约只有 f64）。需要 4 字节浮点请走 v2。
- **NaN 策略差异（有意）**：v1 入口拒绝 NaN；v2 需要在桥接中逐位保留
  NaN 载荷，故 v2 用 `FloatBits(pattern)` 表达 NaN 叶子（有限值仍是
  `float`，Infinity 仍拒绝）。这不是对 v1 的放宽，v1 行为不变。
- **无行为冲突**：v2 布局 A 的普通（非变参、非共享、非联合、非位域、
  ≤64 位、非 f32）调用与 v1 同构；变参保存区等新区域只在用到相应特性
  时才出现在帧中，固定调用与 v1 一致。

# 确定性与“未决”口径

同一输入重复调用必得同一结果（无随机、无主机布局依赖、无全局状态）。
本题不存在需要返回 `unknown` 的数值搜索：所有被支持的输入都有确定
答案，并由独立参考在小参考域完整核对。题面允许的“无解/预算耗尽”以
**具体异常**如实返回且定位到参数路径（重叠位域、标签越界、不可表示的
提升、共享位冲突、值越界、递归类型等），不以 unknown 充当通用实现。

# 验证方式（独立性）

- `tests/test_abi_marshal.py`：原 v1 的 42 个用例（手工字面量 + 文件内
  独立偏移参考），未删除、未放松断言。
- `tests/test_abi_v2.py`：36 个 v2 确定性用例，覆盖两端口、有符号/跨
  单元/重叠位域、空聚合、标签越界、无标签辨识、变参提升与不可表示、
  返回间接区、保存区、共享一致/冲突、32 实参用尽转栈、128 位、NaN
  载荷、桥接原子性、证书肯定/否定判定、确定性。
- `tests/test_abi_v2_extended.py`：9 个扩展用例。**小参考域完整穷举**
  （u8/s8 全部 256/256 值、3×3 有符号位域笛卡尔积、4×4 显式位域、
  带标签联合标签×边界值，均两端口）；随机嵌套（≥6 层、跨单元位域、
  空聚合、联合）编码与完整帧编组差分；A↔B 桥接比较规范值与 NaN 载荷；
  每个随机例子都过独立证书；一个 **32 实参 + i128 + 6 层嵌套 + 共享
  单元 + 变参 + 40 字节间接返回**的主规模非退化端到端例子。
- `tests/reference_v2.py`：**独立逐位参考**，从同一份类型规约 spec 出发
  独立计算布局/编码/分类/寄存器与栈分配，**不调用任何被测布局或编解码
  函数来产生期望值**（只复用类型构造器那是输入对象）。
- `demo_v2.py`：真实计算，展示新增正常结果、一个棘手边界（9 位有符号
  位域 -256/-257）与一次实际触发的拒绝（桥接标签越界、缓冲零改写）。

# 已知限制

- 合成 ABI，不对齐任何真实硬件 ABI，不可直接用于真实 FFI；无 SSEUP/
  X87 等真实规则。
- 位域存储基类限定 1/2/4/8 字节整数（位域宽度 ≤64）；不支持位域数组
  或浮点位域。
- 无标签联合在多成员位模式不可区分时必须给 `tag_hint`，否则如实拒绝。
- 共享单元的视图从单元偏移 0 叠放（不支持任意视图内偏移）。
- `unpack_*` 信任映像内部寄存器/栈结构的一致性（同 v1），但内容正确性
  可由 `verify_image` 独立校验。
