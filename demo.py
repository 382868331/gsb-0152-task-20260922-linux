"""demo.py — abi_marshal 固定输入演示（离线、确定性、无 sleep）。

第 1 部分：v1（LEGACY）旧行为冻结演示；
第 2 部分：v2 新增能力的端到端例子——
  * 显式描述符 A(小端)/B(大端)；
  * 联合体活动成员标签 + 跨存储单元位域 + 可变参数默认提升；
  * 返回值间接区(sret) + 可变参数寄存器保存区 + 按位共享存储单元，
    出现在**同一调用**中；
  * A->B 调用映像桥接：规范值、有符号边界、NaN 原始载荷位；
  * 棘手边界：大端 128 位整数偶数号配对 + 负位域跨字节；
  * 实际触发的拒绝：重叠位域、越界标签桥接、不可表示的提升，
    以及失败时目标缓冲零改写。
"""

import struct

import abi_marshal as am

A, B = am.DESCRIPTOR_A, am.DESCRIPTOR_B


def rec(name, fields):
    r = am.RecordType(name)
    r.define(fields)
    return r


def uni(name, members, tag=am.UINT8):
    u = am.UnionType(name)
    u.define(members, tag_type=tag)
    return u


def part1_legacy():
    print("=" * 70)
    print("第 1 部分：v1 LEGACY（默认入口，行为冻结）")
    print("=" * 70)
    img = am.pack_call([(am.UINT64, 0x1122334455667788),
                        (am.DOUBLE, 2.5),
                        (am.INT32, -7)])
    print(f"  int_regs[0] = {img.int_regs[0]:#018x}")
    print(f"  float_regs[0] = {img.float_regs[0]}")
    print(f"  int_regs[1] = {img.int_regs[1]:#x}（i32 低 4 字节，高位零填）")
    rebuilt = am.unpack_call([am.UINT64, am.DOUBLE, am.INT32], img)
    print(f"  unpack -> {rebuilt}")
    try:
        am.pack_call([(am.UINT8, 300)])
    except am.IntegerOverflowError as exc:
        print(f"  拒绝（旧规则保留）：{type(exc).__name__}: {exc}")
    try:
        am.pack_call([(am.DOUBLE, float("nan"))])
    except am.NonFiniteFloatError as exc:
        print(f"  拒绝（v1 仍拒绝 NaN）：{type(exc).__name__}: {exc}")


def build_types():
    # 跨存储单元的位域：a 5 位、b 10 位（有符号），连续共 15 位 -> 2 字节
    flags = rec("Flags", [("a", am.BitField(5)),
                          ("b", am.BitField(10, True)),
                          ("kind", am.UINT16)])
    # 联合体：标签选择 i32 / f64 / 8 字节数组
    payload = uni("Payload",
                  [("i", am.INT32, 0),
                   ("d", am.DOUBLE, 1),
                   ("blob", am.ArrayType(am.UINT8, 8), 2)])
    # 6 层嵌套：Node{tag, kids[3], w} 逐层包裹联合体
    node = payload
    for d in range(6):
        node = rec(f"Node{d}", [("tag", am.UINT8),
                                ("kids", am.ArrayType(node, 3)),
                                ("w", am.INT32)])
    return flags, payload, node


def make_tree():
    leaf_i = am.Tagged("i", -123456789)
    leaf_d = am.Tagged("d", struct.unpack("<d", struct.pack("<Q",
                       0x7FF0_0000_0000_ABCD))[0])  # 自定义载荷 NaN
    leaf_b = am.Tagged("blob", list(b"ABCDEFGH"))

    def level(d):
        leaves = [leaf_i, leaf_d, leaf_b]
        if d == 0:
            return [{"tag": d, "kids": [leaf_i, leaf_d, leaf_b], "w": -d}
                    for _ in range(1)]
        child = {"tag": d,
                 "kids": [level(d - 1)[0] for _ in range(3)],
                 "w": -1000 - d}
        return [child]

    return level(5)[0]


def part2_v2():
    print()
    print("=" * 70)
    print("第 2 部分：v2 显式描述符（A=小端 6i/4f，B=大端 4i/2f）")
    print("=" * 70)
    flags, payload, node = build_types()

    print("\n-- 2.1 跨存储单元位域：两种相反字节序的物理位 --")
    fv = {"a": 0b10101, "b": -300, "kind": 0xCAFE}
    for desc in (A, B):
        lay = am.layout(flags, abi=desc)
        f = lay.fields[1]
        phys = sorted((s.byte, s.lsb + k, s.value_bit +
                       (s.count - 1 - k if s.rev else k))
                      for s in f.bit_segments for k in range(s.count))
        print(f"  {desc.name}: Flags size={lay.size} 字段'b'物理映射 "
              f"(字节,物理位)<-值位: {phys}")
        data = am.encode(flags, fv, abi=desc)
        back = am.decode(flags, data, abi=desc)
        print(f"    编码={data.hex()}  解码 b={back['b']} (期望 -300)")

    print("\n-- 2.2 联合体标签 + NaN 原始载荷 --")
    nan = struct.unpack("<d", struct.pack("<Q", 0x7FF0_0000_0000_ABCD))[0]
    for desc in (A, B):
        data = am.encode(payload, am.Tagged("d", nan), abi=desc)
        back = am.decode(payload, data, abi=desc)
        got_bits = struct.pack(desc.endian_char + "d", back.value).hex()
        print(f"  {desc.name}: tag={data[0]} 成员={back.member} "
              f"f64 原始位={got_bits}")

    print("\n-- 2.3 同一调用：sret + 保存区 + 共享单元 + 提升 + 位域/联合")
    tree = make_tree()
    ret_t = am.ArrayType(am.UINT8, 20)          # A:20>16, B:20>8 -> sret
    buf_t = am.ArrayType(am.UINT8, 64)
    sig = am.Signature(
        args=(node, am.UINT128, flags, am.PTR),
        varargs=(am.UINT8, am.INT16, am.FLOAT32),   # 默认提升
        ret=ret_t,
        cells=(("buf", buf_t),))
    ret_v = list(range(20))
    cell_v = [(i * 7) & 0xFF for i in range(64)]
    args = [tree, 2 ** 128 - 1, fv, am.CellPtr("buf")]
    vargs = [250, -300, nan]
    pA = am.plan_call(A, sig)
    imgA = am.encode_call(pA, args, vargs, {"buf": cell_v}, ret_v)
    print(f"  A 帧尺寸={pA.frame_size}  返回策略={pA.ret.kind}")
    print(f"  隐藏 sret 指针 -> 槽 {pA.ret.hidden_arg.int_regs}；"
     f"u128 -> 槽 {pA.placements[1].int_regs}")
    print(f"  保存区：整数槽 {pA.save.int_slots} @frame+{pA.save.int_offset}，"
          f"浮点槽 {pA.save.float_slots} @frame+{pA.save.float_offset}")
    print(f"  单元: {[(c.name, c.offset) for c in pA.cells]}")
    nval = sum(1 for e in pA.certificate.entries if e.kind == "value")
    print(f"  证书证据条数={len(pA.certificate.entries)}（其中值位段 {nval}）；"
          f"自检={am.check_certificate(pA.certificate) or '通过'}")
    dA = am.decode_call(pA, imgA)
    print(f"  重建：varargs={dA.varargs}（u8/u16 提升为 i32，f32 提升为 f64）")
    print(f"  重建：第 4 实参={dA.args[3]}（CellPtr 按帧地址重绑定）")
    print(f"  重建：sret 返回前 6 字节={dA.ret[:6]}")

    print("\n-- 2.4 A -> B 桥接（先解码规范值再按 B 编组）--")
    imgB = am.bridge_call(pA, imgA, B)
    dB = am.decode_call(imgB.plan, imgB)
    print(f"  B 返回策略={imgB.plan.ret.kind}；B 帧尺寸={imgB.plan.frame_size}")
    print(f"  B 下 u128 槽号={imgB.plan.placements[1].int_regs}"
          f"（隐藏 sret 指针已占槽 0；B 要求偶数号配对 -> 从槽 2 起）")
    print(f"  桥接后 Flags.b={dB.args[2]['b']}（期望 -300，有符号边界保留）")
    leaf = dB.args[0]
    for _ in range(6):
        leaf = leaf["kids"][0]
    detail = (struct.pack(">d", leaf.value).hex() if leaf.member == "d"
              else leaf.value)
    print(f"  深层联合叶子(6 层下钻): 成员={leaf.member} 值/载荷={detail}")
    f32bits = struct.pack("<f", dB.varargs[2]).hex()
    print(f"  桥接后 f32 可变参数的 binary32 位={f32bits}（先入 f32 域再提升）")
    print(f"  桥接后返回值前 6 字节={dB.ret[:6]}；单元前 4 字节={dB.cells['buf'][:4]}")

    print("\n-- 2.5 棘手边界：寄存器用尽整体转栈（不消耗任何寄存器）--")
    pB3 = am.plan_call(B, am.Signature(
        args=(am.INT64, am.INT64, am.INT64,
              rec("TwoI", [("a", am.INT64), ("b", am.INT64)]),
              am.DOUBLE)))
    print(f"  B 只有 4 个整数槽：3 个 i64 后，两整块聚合安放="
          f"{pB3.placements[3].kind}，消耗槽={pB3.placements[3].int_regs}")
    print(f"  其后 double 仍拿到浮点槽：{pB3.placements[4].kind}")

    print("\n-- 2.6 实际触发的拒绝 --")
    # (a) 重叠位域，错误定位到字段路径
    bad = am.RecordType("Bad")
    bad.define([("x", am.BitField(6)), ("y", am.BitField(4), 4)])
    try:
        am.layout(bad, abi=A)
    except am.BitFieldError as exc:
        print(f"  (a) {type(exc).__name__}: {exc}")

    # (b) 越界标签：先构造一个联合体走栈的调用，篡改其标签字节再桥接
    fill = [(am.INT64, i) for i in range(6)]   # A 的 6 个整数槽占满
    ps = am.plan_call(A, am.Signature(
        args=tuple([am.INT64] * 6 + [payload])))
    ims = am.encode_call(ps, [v for _t, v in fill] +
                         [am.Tagged("i", 7)])
    frame = bytearray(ims.frame)
    tag_off = ps.placements[6].frame_offset     # 联合体标签在栈起点
    frame[tag_off] = 99
    bad_img = am.V2CallImage(
        ims.int_regs, ims.float_regs, bytes(frame),
        ims.placements, ims.vararg_placements,
        ims.ret_int, ims.ret_float, ps)
    try:
        am.bridge_call(ps, bad_img, B)
    except am.BridgeError as exc:
        print(f"  (b) {type(exc).__name__} reason={exc.reason}: "
              f"{exc.path}: 标签 99 不对应任何成员")

    # (c) 不可表示的提升：128 位整数经可变参数传递
    try:
        am.plan_call(A, am.Signature(varargs=(am.UINT128,)))
    except am.PromotionError as exc:
        print(f"  (c) {type(exc).__name__}: {exc}")

    # (d) 桥接失败不允许部分改写目标缓冲
    no_span = am.AbiDescriptor(
        name="X", endian="big", int_reg_count=4, float_reg_count=2,
        max_reg_bytes=16, align_int={1: 1, 2: 2, 4: 4, 8: 8, 16: 16},
        bitfield_span=False, i128_even_pair=True, sret_threshold=8)
    sigx = am.Signature(args=(flags,))
    px = am.plan_call(A, sigx)
    ix = am.encode_call(px, [fv])
    dst = bytearray(b"\x5A" * 32)
    before = bytes(dst)
    status, plan = am.bridge_into(px, ix, no_span, dst)
    print(f"  (d) 桥接到禁止跨单元的布局：status={status}，"
          f"目标缓冲被改写={bytes(dst) != before}（必须为 False）")


def main():
    part1_legacy()
    part2_v2()
    print()
    print("demo 完成（全部输出均为现场计算，无预录答案、无 sleep）。")


if __name__ == "__main__":
    main()
