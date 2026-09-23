# abi_marshal — 聚合参数的合成 ABI 编组

离线外部函数接口（FFI）工程的可复用纯 Python 模块：把 `(类型, 值)` 实参
编组为“寄存器 + 栈帧”调用映像，并可从映像重建参数。当前包含两代契约：

* **v1（LEGACY，默认入口）**：上一版的小端契约，字节行为完全冻结；
* **v2（显式描述符）**：两种相反字节序布局、联合体标签、跨存储单元位域、
  可变参数默认提升、返回值间接区、可变参数寄存器保存区、按位共享存储
  单元、逐位布局证书，以及两种布局间的调用映像桥接。

两代共享同一类型数据模型；v2 是 v1 的严格超集（v1 入口不接受 v2 专属
类型），**没有另起的平行项目**。

## 运行环境

- Python **3.14.7**，仅标准库，无需安装任何依赖；
- Linux 原生、全离线：不需要网络、账号、密钥、数据库、容器、硬件或第三方
  API；不读取环境变量、配置备份或凭据；**不调用本机 FFI / `ctypes`/`struct
  ` 的主机原生布局推导**（`struct` 仅用于定宽整数与 IEEE-754 浮点的位模式
  转换）；
- 所有示例数据由代码确定性生成，可重复运行得到同一结果。

## 命令与预算

```sh
# 快速回归（v1 全部旧测试 42 个 + v2 快速测试，约 0.2 秒）
python -m unittest discover -s tests -v

# 扩展验证（另加：小域完整枚举、120 组随机嵌套双端桥接、主规模大例子；
# 约 1 秒。默认跳过，不拖慢快速回归）
ABI_EXTENDED=1 python -m unittest discover -s tests -v

# 固定输入演示（约 0.2 秒，远低于 90 秒预算；无 sleep、无预录答案）
python demo.py
```

| 预算项 | 值 | 含义 |
|---|---|---|
| `DEFAULT_BUDGET` | 20,000,000 | `plan_call` 工作单位（按类型/帧位计费），耗尽抛 `BudgetExhausted` |
| v2 定位参数上限 | 32（`max_args`，可在描述符配置） | 超出抛 `InvalidTypeError("args")` |
| v2 可变参数上限 | 32（`max_varargs`） | 同上 |
| 类型总字节上限 | `MAX_TYPE_SIZE = 1 MiB` | v1/v2 共用 |
| 数组元素上限 | 65536；记录/联合成员上限 256 | |
| 位域位宽 | 1..64；存储单元固定 8 位（字节） | |
| 整数宽度 | 1/2/4/8（v1、v2）+ 16 字节 128 位（仅 v2） | |

预算耗尽（`BudgetExhausted`）与“已证明无解”的确定性拒绝（`BitFieldError`/
`TagError`/`PromotionError`/`BridgeError`）是**不同的异常类型**，绝不混用。

---

## v1 契约（LEGACY，默认，行为冻结）

- 小端；指针 8 字节无符号；整数 1/2/4/8 字节；浮点仅 binary64；
- 记录自然对齐、对齐上限 8；空记录 `size=0, align=1`；数组连续满尺寸排列；
- 聚合按 8 字节块分类：块内含任意整数/指针字节 → `INTEGER`，否则 `FLOAT`；
  超过 16 字节 → `MEMORY` 整体走栈；
- 整数寄存器 4、浮点寄存器 2；聚合寄存器不足时**整体**走 16 字节对齐栈，
  **不消耗任何寄存器**；空记录不占任何资源；
- 入口拒绝 NaN/Infinity、bool 冒充整数、int 冒充 float。

v1 公开入口签名未变：`layout/classify/encode/decode/pack_call/unpack_call`。

---

## v2 契约（显式描述符）

### 1. 描述符：目标布局的唯一事实来源

`AbiDescriptor` 显式声明目标布局，新布局**只能**由描述符推导，不查主机
C 结构：

| 字段 | 说明 | A-little (`DESCRIPTOR_A`) | B-big (`DESCRIPTOR_B`) |
|---|---|---|---|
| `endian` | `"little"` / `"big"` | little | **big** |
| `int_reg_count` / `float_reg_count` | 寄存器槽数 | 6 / 4 | 4 / 2 |
| `stack_align` | 栈帧各区域最小对齐 | 16 | 16 |
| `max_reg_bytes` | 聚合经寄存器传递的尺寸上限 | 32 | 16 |
| `max_agg_align` | 聚合对齐上限（超出截断） | 16 | 16 |
| `align_int` / `align_float` / `align_ptr` | 标量对齐表 | 1/2/4/8/16 字节→1/2/4/8/**8** | 同左但 16→**16** |
| `bitfield_span` | 位域可否跨存储单元 | True | True（另有内置 `bitfield_span=False` 的拒绝用例） |
| `i128_even_pair` | 128 位整数两槽是否要求偶数号起配对 | False | **True** |
| `sret_threshold` | 返回值超过此尺寸走隐式 sret | 16 | 8 |

别名：`V2_LE = DESCRIPTOR_A`，`V2_BE = DESCRIPTOR_B`。

### 2. 类型（在 v1 同一数据模型上扩展）

- `IntNType(16, signed)` / 常量 `INT128`、`UINT128`；
- `Float32Type()` / 常量 `FLOAT32`（binary32）；
- `BitField(width, signed)`：**只能**作为记录成员。连续位域成员组成一个
  位域段，线性位地址按“值的最低位方向”递增；物理映射（合成 ABI 明示）：
  - 小端：线性位 g → 字节 `g//8` 的位 `g%8`；
  - 大端：线性位 g → 字节 `(run_bytes-1-g//8)` 的位 `(7-g%8)`。
  - 单个位域可跨字节（`bitfield_span=True` 时），`FieldLayout.bit_segments`
    逐段给出物理位置（`byte/lsb/count/value_bit/rev`），这就是逐位布局证据；
  - 记录字段第三项可给 `pin`（段内显式线性位偏移）。任何区间重叠抛
    `BitFieldError`，错误定位到 `record 'R' field 'f'`，经规划器时带参数
    路径如 `arg[2].hdr.flags: record 'R' field 'f'`；描述符禁止跨越时抛
    `BitFieldError`。
- `UnionType(name).define([(成员名, 类型, 标签值), ...], tag_type=UINT8)`：
  布局为 `[tag][storage]`，`storage` 起始对齐到联合体对齐，尺寸取最大成员，
  整体向上取齐。值为 `Tagged(成员名, 值)`；标签值重复/越界在类型定义期拒绝；
  编码时成员名不存在抛 `TagError`；**解码时标签不对应任何成员抛 `TagError`**
  （越界标签），定位到该参数/字段路径。联合体按标签 + 所有成员字节的静态
  并集参与 8 字节块分类（保守但确定）。
- 联合体成员不能是裸 `BitField`（用一层记录包裹）。

### 3. 可变参数默认提升（明示规则）

- `<4` 字节整数（u8/i8/u16/i16）→ 同符号性 32 位整数；
- `f32` → `f64`：编码时先舍入到 binary32 规范值域，再以 binary64 承载；
  解码时还原为 f32。NaN 的 binary32 载荷位被保留；
- 128 位整数经可变参数传递无提升目标 → `PromotionError`（路径
  `vararg[i]`）。
- `Plan.promoted[i]` 记录每个可变参数的提升目标（未提升为 `None`）。

### 4. 帧与寄存器（规划器 `plan_call`，不接触任何值）

帧从偏移 0 起依次排布（每段起始按 `stack_align` 与自身对齐取齐）：

1. **隐式 sret 隐藏指针**（仅当返回值走 sret）：逻辑首参数，先于用户参数
   分配寄存器/栈；
2. 定位参数；
3. 可变参数（寄存器分配延续同一计数；走栈的同样在帧内）；
4. **共享单元区**：`Signature(cells=...)` 声明的命名单元，外加隐式 `$sret`
   单元；
5. **可变参数寄存器保存区**：凡进入寄存器的可变参数，其所用整数/浮点槽在
   帧内各有一段按位拷贝（`SaveArea.int_offset/float_offset/int_slots/
   float_slots`）；
6. 尾部填充到 `stack_align`。

寄存器槽语义（两种字节序统一）：槽是 8 字节块的**内存别名**，块数据在槽
低 n 字节（块内自然偏移），槽的 64 个原始位按物理 little 位序编号为整数；
窄于 8 字节一律零扩展（有符号性由整数类型保留，不做槽内符号扩展）。
`V2CallImage` 的浮点槽同样存原始 64 位（NaN 载荷无损），用 `struct` 按
f64/f32 解释。128 位整数占两个连续整数槽（块 0→低槽），描述符 B 要求偶数
号起配对：奇数槽留空且**不回溯**给其他实参。整体回退规则同 v1：放不下
（含配对落空）就整体走栈，不消耗寄存器。

### 5. 返回值与共享单元

- 返回类型尺寸 ≤ `sret_threshold` 且分类不为 MEMORY → 寄存器返回
  （`ret_int/ret_float` 原始位，按分类顺序）；
- 否则 → 隐式 sret：帧内建立 `$sret` 单元，隐藏首参数为指向它的指针；
- 或在 `Signature(ret_cell="out")` 中显式指定已声明单元作为返回载体
  （该单元类型必须与 `ret` 是同一个 Type 对象）。
- 指针实参可给 `CellPtr("名字")`：编码时回填为
  `FRAME_BASE + 单元帧偏移`（`FRAME_BASE = 0x7F0000000000`，合成相对地址
  空间），解码时按偏移重绑定回 `CellPtr`；不指向任何已声明单元的地址按
  普通整数指针保留。同一单元因此可被“指针实参”和“返回区/单元内容”按位
  共享，证书中以 `shared_by` 标注。

### 6. 逐位证书

`plan_call` 返回的 `Plan.certificate` 是 `Certificate(entries, frame_bits,
int_regs, float_regs)`；`entries` 是 `BitEvidence(region, start, length,
kind, path, src_bit, src_region, shared_by)`：

- `region`：`ireg:k` / `freg:k`（槽内 64 位）或 `frame`（帧内位）；
- `kind`：`value`（参数/字段值位，带参数路径与值内位索引）、`tag`（联合体
  标签位）、`union`（联合体共享存储位，活动成员由标签运行时选择）、
  `save-copy`（保存区位，`src_region` 指向源寄存器槽）、`cell`/`ret`
  （共享单元 / 返回载体）、`pad`（确定性零填充）；
- 不变量：**每个寄存器槽 64 位与帧的每个有效位都被恰好一条证据覆盖，不重
  不漏**。`check_certificate()` 是规划器自带自检；验收用的独立检查器
  （`tests/reference.py`）从类型与描述符重新推导，不读取该函数结论，也不
  信任求解器输出的任何布尔标志。
- 重放到小端与大端描述符后，独立检查器还复核：所有栈参数偏移、单元偏移、
  保存区与帧尺寸满足各自对齐约束，返回值策略与阈值一致。

### 7. A ↔ B 桥接

`bridge_call(src_plan, src_image, dst_desc)` /
`bridge_into(..., bytearray)`：

1. `decode_call` 解码为**规范值**（Python 表示 + 标签选择的活动成员）；
2. 按目标描述符 `plan_call` 重新规划；
3. `encode_call` 重新编组。

保留语义：有符号边界（负位域、负 i128 等）按类型重新二补码编码；NaN 按
原始载荷位跨域保留（f32 可变参数在 binary32 子域内保留载荷）。比较标准是
**规范值相等 + 约定保留位相等**（测试中 NaN 逐位比较），**不把填充字节
相等当作判据**——填充位定义为零约定，但即使源映像槽内填充位被改脏，
桥接只搬运值位，规范值不变（测试 `test_bridge_does_not_require_padding_
equality` 专门覆盖）。

失败全部在写入任何目标状态前抛 `BridgeError`，`reason ∈ {"value",
"promotion", "tag", "unidentifiable", "budget"}`；`bridge_into` 先在临时
映像构造成功后才整体拷贝进调用方缓冲，失败时缓冲**一个字节都不改**。

### 8. v2 值约定与错误

- 整数/指针仍拒绝 `bool`；v2 **允许** NaN/±Infinity（与 v1 的区别，见迁移
  说明）并保留其位模式；f64 只接受 `float`；
- **具名 f32 参数/字段**要求值已能被 binary32 精确表示（否则抛
  `InvalidValueError`，不做静默舍入）；需要舍入语义时请显式先做
  `struct.pack/unpack("f")`。仅**可变参数 f32** 按默认提升规则统一舍入到
  f32 域后再以 f64 承载（这是规则本身，不是静默纠正）；
- 记录 dict 键集合必须精确匹配；数组长度必须匹配；
- 错误层级：`ABIError` → `InvalidTypeError`/`InvalidValueError`；
  `RecursiveTypeError ⊂ InvalidTypeError`（记录/联合/数组循环，直接间接
  均检测）、`IntegerOverflowError`、`BitFieldError ⊂ InvalidTypeError`、
  `TagError ⊂ InvalidValueError`、`PromotionError ⊂ InvalidValueError`、
  `BudgetExhausted`、`BridgeError`。所有错误携带 `path`。

---

## 公开接口

| 入口 | 代 | 语义 |
|---|---|---|
| `layout(t)` / `layout(t, abi=desc)` | v1/v2 | 字节布局（v2 含位域段、联合成员） |
| `classify(t)` / `classify(t, abi=desc)` | v1/v2 | 8 字节块分类 |
| `encode/decode(..., abi=desc)` | v1/v2 | 类型化字节编解码 |
| `pack_call/unpack_call` | v1 | 旧寄存器+栈映像（冻结） |
| `AbiDescriptor`, `DESCRIPTOR_A/B` | v2 | 目标布局声明 |
| `Signature(args, varargs, ret, cells, ret_cell)` | v2 | 静态调用类型 |
| `plan_call(desc, sig, *, budget=...)` | v2 | 纯静态规划 → `Plan`（含 `Certificate`） |
| `encode_call(plan, argv, varargv, cells, ret_value)` | v2 | 规划与编码分离；全成功才出映像 |
| `decode_call(plan, image)` | v2 | → `CallValues(args, varargs, cells, ret)` |
| `bridge_call/bridge_into` | v2 | 两布局映像转换 |
| `check_certificate(cert)` | v2 | 证书自检（独立检查器不依赖它） |

Python 值表示：整数/指针 `int`（或 `CellPtr`）；浮点 `float`；数组 `list`；
记录 `dict`；联合 `Tagged(member, value)`；空记录 `{}`；空数组 `[]`。

## 确定性

无全局状态、无随机源、无网络/时钟依赖；同一 `(描述符, 签名, 值)` 重复调用
得到字节完全相同的映像（测试 `test_deterministic_repetition`）。唯一的
“未决”出口是显式 `BudgetExhausted`（调大 `budget` 可消除）；小规模独立
参考覆盖的例子一律给确定答案，不用 unknown 搪塞。

## 与 v1 的差异及迁移办法

v1 调用方**无需任何改动**（42 个旧测试一字未删、断言未放松，全部通过）。
逐项差异：

1. **v2 才有的类型**（`INT128/UINT128/FLOAT32/BitField/UnionType`）在 v1
   入口抛 `InvalidTypeError`。迁移：改用 `layout(t, abi=...)` 与
   `plan_call/encode_call`。
2. **NaN/Infinity**：v1 仍在 `encode/pack_call` 抛 `NonFiniteFloatError`；
   v2 允许并保留原始位。需要旧拒绝行为时继续用 v1 入口。
3. **新功能走显式选项**：新布局必须用 `AbiDescriptor` 显式声明；不传
   `abi=` 的一切调用逐字节等于旧版。
4. `pack_call` 的参数上限维持 v1 的 1024；32 个参数的范围要求由 v2
   `Signature` 在描述符 `max_args=32` 下满足（扩展测试实际构造 32 参数）。

## 验证方式（参考不重复调用被测求解器）

- `tests/test_abi_marshal.py`：**v1 原 42 个测试，原样保留**；
- `tests/test_v2.py`：40 个快速 v2 用例 + `tests/reference.py` 中的
  **独立平行参考**（自写布局/编码/分类/规划/映像重放/证书核对，只消费
  类型声明字段，从不调用 `abi_marshal` 取期望值）；含硬编码字面量核对
  （物理位、标签字节、槽号、帧偏移）；
- `tests/test_extended.py`（`ABI_EXTENDED=1`）：
  - 小参考域完整枚举：位域两字段宽度 (1..8)² × 两端序 × 全部符号边界；
    全部标量两两组合（含 i128 奇偶配对）的安放分类；
  - 120 组固定种子随机嵌套聚合（≥6 层，记录/数组/联合/位域混合），覆盖
    大小端、负位域、空聚合、寄存器用尽转栈；每组独立逐位核对映像与证书，
    并双向桥接后比较规范值与保留位；
  - 主规模非退化例子：6 层 3 路树（叶子为联合，>14 KB 聚合）+ 32 个参数 +
    可变参数 + sret + 64 字节共享单元，帧 >17 KB，A/B 双向桥接来回一致。
- 桥接测试比较规范值与约定保留位（NaN 按位、有符号边界），明确
  **不**以填充字节相等为唯一标准；对认证/覆盖结论，检查器从类型重新推导，
  不信任求解器的布尔标志。

## 已知限制

- 是题面定义的合成 ABI，不兼容 System V AMD64 / AAPCS64 等真实平台 ABI，
  不能直接发起真实 FFI 调用；
- 保存区是“槽原始位的逻辑位映像”（与数据字节序无关的逐位拷贝），不是某
  个真实平台的 vararg save area 内存格式；
- 位域存储单元固定 8 位（字节），位宽上限 64；位域段内不做真实编译器的
  “同类型合并分配单元”启发式，段布局由线性位规则完全确定；
- f32 可变参数的合法规范值是“能被 binary32 精确表示的 float”，编码时
  统一舍入；桥接不替调用方保留一个不可由 f32 表示的 f64 有效数字（该选择
  是默认提升规则的必然结果）；
- v1 `unpack_call` 与 v2 `decode_call` 信任映像的结构一致性（槽/偏移由
  Plan 给出），只对帧截断、空槽等做结构性校验，不模拟恶意破坏的映像；
- 指针重绑定只识别指向已声明帧单元（`FRAME_BASE+offset`）的地址，其余
  地址按不透明整数指针原样跨桥接传递（不做指针所指内存的深拷贝）。
