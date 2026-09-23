"""demo_v2.py — 双布局 / 联合 / 位域 / 变参 / 桥接 / 证书 固定输入演示。

离线、确定性、无 sleep、不预录结果，约 1 秒内真实计算完成。一次调用同时
用到本题三组新能力与补充要求：

* 联合体活动成员标签（带标签联合，标签随布局走）；
* 位域跨存储单元（9 位有符号位域，含负边界值 -256）；
* 可变参数默认提升（f32->f64、i16->i32）及变参寄存器保存区；
* 同一调用中的返回值间接区（40 字节聚合）与按位共享存储单元（双视图）；
* 128 位整数、NaN 原始载荷位；
* A(小端)->B(大端)->A 调用映像桥接，比较规范值而非填充字节；
* 证书独立重算（不信任任何自报布尔标志）；
* 一个棘手边界与一次实际触发的拒绝（桥接标签越界，目标缓冲零改写）。

运行：python demo_v2.py
"""

import abi_marshal as am


def hr(title):
    print(f"\n== {title} " + "=" * (66 - len(title)))


def build_types():
    # 跨存储单元的位域记录：两个 9 位有符号位域共享 u16 单元流，
    # 第二个位域从第 9 位起，跨越 16 位单元边界。
    regs = am.RecordType("Regs")
    regs.define([("lo", am.BitField("lo", am.UINT16, 9)),
                 ("hi", am.BitField("hi", am.INT16, 9))])

    # 带标签联合：要么是一个 i32，要么是上面的位域记录。
    word = am.UnionType("Word")
    word.define([("num", am.INT32), ("regs", regs)])

    # 共享存储单元的两个视图：一个 u16 整词，一个 4+12 位拆分。
    whole = am.RecordType("Whole")
    whole.define([("w", am.UINT16)])
    parts = am.RecordType("Parts")
    parts.define([("nibble", am.BitField("nibble", am.UINT16, 4)),
                  ("high", am.BitField("high", am.UINT16, 12))])

    # 40 字节返回聚合（超过两端口的寄存器聚合界 16/32 -> 返回间接区）。
    block = am.RecordType("Block")
    block.define([("bytes", am.ArrayType(am.UINT8, 40))])
    return regs, word, whole, parts, block


def main():
    regs, word, whole, parts, block = build_types()

    hr("1. 描述符：两种相反字节序布局（均显式声明，不靠主机推导）")
    for d in (am.LAYOUT_A, am.LAYOUT_B):
        r = d.int_regs
        print(f"  {d.name}: endian={d.endian:6s} max_align={d.max_align:2d} "
              f"int_regs={r.int_count} float_regs={r.float_count} "
              f"agg<={r.max_aggregate_bytes}B stack_align={d.stack_align}")

    hr("2. 位域布局证据（跨存储单元；两种字节序）")
    for d in (am.LAYOUT_A, am.LAYOUT_B):
        lay = am.layout_v2(regs, d)
        print(f"  [{d.name}] Regs size={lay.size} align={lay.align}")
        for f in lay.fields:
            for s in f.segs:
                print(f"      {s.name:4s} 字节+{s.byte} 位{s.lsb:<2d} "
                      f"宽{s.width}  ({'跨单元段' if s.width < 9 and s.name in ('hi',) and s.byte else ''})")

    # 一次调用的实参规约
    params = [
        am.Param(word),                                   # 带标签联合
        am.Param(regs),                                   # 跨单元位域
        am.Param(am.FLOAT32, variadic=True),              # f32 -> f64
        am.Param(am.INT16, variadic=True),                # i16 -> i32
        am.Param(am.INT128),                               # 128 位整数
        am.Param(am.DOUBLE),                               # NaN 载荷
        am.Param(whole, shared="io", shared_size=2),       # 共享视图 A
        am.Param(parts, shared="io", shared_size=2),       # 共享视图 B
    ]
    vals = [
        am.UnionValue("regs", {"lo": 511, "hi": -256}),   # 无符号最大 + 负边界
        {"lo": 300, "hi": -256},                           # 跨单元 + 负位域
        1.5,
        -2,
        (1 << 127) - 1,
        am.FloatBits(0x7FF8000000000ABC),                  # 静默 NaN + 自定义载荷
        None,
        None,
    ]
    shared = [am.SharedUnit("io", 2, {
        "Whole": {"w": 0xBEEF},
        "Parts": {"nibble": 0xF, "high": 0xBEE}})]
    ret_value = {"bytes": list(range(40))}

    hr("3. 布局 A（小端）编组：寄存器 / 返回间接区 / 保存区 / 帧")
    imgA = am.pack_call_v2(params, vals, am.LAYOUT_A,
                          ret_type=block, ret_value=ret_value,
                          shared_units=shared)
    for i, (p, v) in enumerate(zip(params, vals)):
        pl = imgA.plan.placements[i]
        extra = (f" int_regs={pl.int_regs} float_regs={pl.float_regs}"
                 if pl.kind == "registers" else
                 f" stack@{pl.stack_offset}" if pl.kind == "stack" else
                 f" alias->arg[{pl.alias_of}]" if pl.kind == "alias" else "")
        print(f"  arg[{i}] {p.type.name:8s} variadic={p.variadic!s:5s} "
              f"-> {pl.kind}{extra}")
    ri = imgA.plan.return_info
    print(f"  return: {'间接区 @0, %d 字节' % ri.area_size if ri.indirect else '寄存器'}")
    saves = [(e.kind, e.slot) for e in imgA.plan.save_entries]
    print(f"  variadic save entries={saves} frame_size={imgA.plan.frame_size}")
    print(f"  int_regs={[hex(x) if isinstance(x, int) else x for x in imgA.int_regs]}")

    hr("4. 独立证书：覆盖每一有效位、不重叠，且反字节序重放仍满足对齐")
    fc = am.certify_frame(params, imgA.plan, am.LAYOUT_A)
    ic = am.verify_image(params, vals, imgA, am.LAYOUT_A,
                         shared_units=shared, ret_value=ret_value)
    print(f"  frame certificate : valid={fc.valid} "
          f"regions_disjoint={not fc.region_overlaps} "
          f"save_ok={fc.save_ok} flipped_endian_replay={fc.flipped_align_ok}")
    print(f"  image certificate : valid={ic.valid} "
          f"evidence_bits={ic.evidence_bits}/{ic.data_bits} "
          f"content_mismatches={ic.content_mismatches}")
    am.certify_layout(word, am.LAYOUT_A).raise_if_invalid()
    am.certify_layout(regs, am.LAYOUT_B).raise_if_invalid()
    print("  layout certificates for union/bitfields (A 与 B): valid")

    hr("5. A -> B -> A 桥接：先解码规范值再编组（比较规范值，不比较填充）")
    imgB = am.bridge_call(imgA, params, am.LAYOUT_A, am.LAYOUT_B,
                         ret_type=block, shared_units=shared)
    imgA2 = am.bridge_call(imgB, params, am.LAYOUT_B, am.LAYOUT_A,
                          ret_type=block, shared_units=shared)
    outB = am.unpack_call_v2(params, imgB, am.LAYOUT_B)
    outA = am.unpack_call_v2(params, imgA2, am.LAYOUT_A)

    def canonical(out):
        """语义比较口径：联合标签归一为成员下标；共享占位取 SharedUnit 视图。"""
        norm = []
        views = dict(shared[0].views)   # {类型名: 视图值}
        for i, (p, x) in enumerate(zip(params, out)):
            if isinstance(x, am.UnionValue):
                idx = x.tag if isinstance(x.tag, int) else \
                    [n for n, _ in word.members].index(x.tag)
                x = am.UnionValue(idx, x.value)
            if p.shared is not None:
                x = views[p.type.name]
            norm.append(x)
        return norm

    expected = [am.UnionValue(1, vals[0].value)] + vals[1:6] + \
        [shared[0].views["Whole"], shared[0].views["Parts"]]
    print("  B 布局 endian =", imgB.plan.desc.endian,
          " frame_size =", imgB.plan.frame_size)
    print("  arg[0] 联合(标签+活动成员) =", outB[0])
    print("  arg[1] 跨单元位域           =", outB[1],
          "(hi=-256 有符号边界保持)")
    print("  arg[2] f32 变参提升为 f64   =", outB[2])
    print("  arg[3] i16 变参提升为 i32   =", outB[3])
    print(f"  arg[4] i128                 = {outB[4]} (== 2**127-1 ? "
          f"{outB[4] == (1 << 127) - 1})")
    print("  arg[5] NaN 原始载荷位       =", outB[5])
    print("  arg[6]/[7] 共享单元两视图   =", outB[6], outB[7])
    print("  返回值(间接区) B 侧重建     =",
          am.unpack_return_v2(block, imgB, am.LAYOUT_B) == ret_value)
    print("  B 与回桥 A 的规范值均等于输入:",
          canonical(outB) == expected and canonical(outA) == expected)

    hr("6. 棘手边界：9 位有符号位域取 -256 / 511 之外即拒绝")
    boundary = am.RecordType("B9")
    boundary.define([("s9", am.BitField("s9", am.INT16, 9))])
    print("  encode(s9=-256) =", am.encode_v2(boundary, {"s9": -256},
                                              am.LAYOUT_A).hex(),
          "-> decode", am.decode_v2(boundary,
              am.encode_v2(boundary, {"s9": -256}, am.LAYOUT_A),
              am.LAYOUT_A)["s9"])
    try:
        am.encode_v2(boundary, {"s9": -257}, am.LAYOUT_B)
        print("  ERROR: -257 未被拒绝")
    except am.IntegerOverflowError as exc:
        print(f"  -257 被拒绝: {type(exc).__name__}: {exc}")

    hr("7. 实际触发的拒绝：桥接源映像标签越界 -> BridgeError，缓冲零改写")
    # arg[0] 是带标签联合；找到其首块所在整数寄存器，把标签字节改成越界值 99
    pl0 = imgA.plan.placements[0]
    tag_slot = pl0.int_regs[0]
    bad_regs = list(imgA.int_regs)
    bad_regs[tag_slot] = (bad_regs[tag_slot] & ~0xFF) | 99  # 小端：标签在最低字节
    tampered = am.CallImageV2(
        imgA.plan, tuple(bad_regs), imgA.float_regs, imgA.frame,
        imgA.evidence, imgA.return_regs, imgA.return_area)
    dst = bytearray(b"\x5a" * 160)
    before = bytes(dst)
    try:
        am.bridge_call_into(dst, tampered, params, am.LAYOUT_A, am.LAYOUT_B,
                            ret_type=block, shared_units=shared)
        print("  ERROR: 越界标签未触发拒绝")
    except am.BridgeError as exc:
        print(f"  触发 BridgeError reason={exc.reason!r} path={exc.path}")
        print(f"  目标缓冲 160 字节保持原样: {bytes(dst) == before}")

    print("\ndemo_v2 完成。")


if __name__ == "__main__":
    main()
