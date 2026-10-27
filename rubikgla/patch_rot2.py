# -*- coding: utf-8 -*-
"""对已提交 S0 版 delta_triton.py 加 return_state + dS0/ds_last (重写版, 精确匹配已提交内容)."""
import io
p = r"F:\夸克\rubikgla\delta_triton.py"
t = io.open(p, encoding="utf-8").read()
miss = []
def rep(old, new, tag):
    global t
    if old not in t:
        miss.append(tag); return
    t = t.replace(old, new, 1)

# 1) fwd: SOUT + RETURN_STATE
rep("    K, V, RATE, ETA, Q, GROT, S0, OUT, SSTORE,\n",
    "    K, V, RATE, ETA, Q, GROT, S0, OUT, SSTORE, SOUT,\n", "f1")
rep("HAS_S0: tl.constexpr,\n):", "HAS_S0: tl.constexpr, RETURN_STATE: tl.constexpr,\n):", "f2")
t2 = t
io.open(p + ".bak", "w", encoding="utf-8").write(t)
# 手工找 fwd 里的读出段末尾 (唯一的 static_range 块在 fwd; bwd 用的是别的变量名)
old_rd = """        for m in tl.static_range(KVG):
            qm = tl.load(q_base + t * H * DH + m * DH + offs_i).to(tl.float32)
            ym = tl.sum(qm[:, None] * S, axis=0)
            tl.store(o_base + t * G * KVG * DH + m * DH + offs_js,
                     ym.to(OUT.dtype.element_ty))"""
rep(old_rd, old_rd + """
    if RETURN_STATE:
        so_base = SOUT + (b * G + g) * DH * DH
        tl.store(so_base + offs_i[:, None] * DH + offs_js[None, :], S.to(SOUT.dtype.element_ty))""", "f3")

# 2) bwd: DSL/DS0 指针与 constexpr
rep("    K, V, RATE, ETA, Q, GROT, DY, SSTORE, DK, DV, DRATE, DETA, DQ, DGROT,\n",
    "    K, V, RATE, ETA, Q, GROT, DY, SSTORE, DK, DV, DRATE, DETA, DQ, DGROT, DSL, DS0,\n", "b1")
rep("STAGES: tl.constexpr, HAS_ROT: tl.constexpr,\n):", "STAGES: tl.constexpr, HAS_ROT: tl.constexpr, HAS_DSL: tl.constexpr, HAS_DS0: tl.constexpr,\n):", "b2")
rep("""    dS = tl.zeros((DH, VH), dtype=tl.float32)

    for _i in tl.range(0, L, num_stages=STAGES):""",
    """    dS = tl.zeros((DH, VH), dtype=tl.float32)
    if HAS_DSL:
        dsl_base = DSL + (b * G + g) * DH * DH
        dS = dS + tl.load(dsl_base + offs_i[:, None] * DH + offs_jl[None, :]).to(tl.float32)

    for _i in tl.range(0, L, num_stages=STAGES):""", "b3")
rep("""        dS = gt1 + kt[:, None] * dr[None, :]""",
    """        dS = gt1 + kt[:, None] * dr[None, :]
        if HAS_DS0:
            tl.atomic_add(ds0_base + (b * G + g) * DH * DH + offs_i[:, None] * DH + offs_jl[None, :], dS)""", "b4")

# 3) Function forward
rep("    def forward(ctx, k, v, rate, eta, q, grot, s0, kv, slots, kvg):",
    "    def forward(ctx, k, v, rate, eta, q, grot, s0, kv, slots, kvg, return_state=False):", "F1")
rep("""        out = torch.empty(B, L, G, kvg, DH, device=k.device, dtype=torch.bfloat16)
        sstore = torch.empty(B, L, G, DH, DH, device=k.device, dtype=torch.float32)
        grid = (B * G * VSPLIT,)
        delta_fwd_kernel[grid](k, v, rate, eta, q, grot, s0, out, sstore,
                               L=L, G=G, KV=kv, H=H, DH=DH, VH=VH, VSPLIT=VSPLIT, KVG=kvg, STAGES=STAGES, HAS_ROT=HAS_ROT_, HAS_S0=HAS_S0_)
        ctx.save_for_backward(k, v, rate, eta, q, grot, sstore)
        ctx.kv, ctx.kvg = kv, kvg
        return out""",
    """        needs = s0 is not None and s0.requires_grad
        out = torch.empty(B, L, G, kvg, DH, device=k.device, dtype=torch.bfloat16)
        sstore = torch.empty(B, L, G, DH, DH, device=k.device, dtype=torch.float32)
        sout = (torch.empty(B, G, DH, DH, device=k.device, dtype=torch.bfloat16) if return_state
                else torch.empty(1, device=k.device, dtype=torch.bfloat16))
        ds0 = (torch.zeros(B, G, DH, DH, device=k.device, dtype=torch.float32)
               if (ctx.has_s0 and needs) else torch.empty(1, device=k.device, dtype=torch.float32))
        grid = (B * G * VSPLIT,)
        delta_fwd_kernel[grid](k, v, rate, eta, q, grot, s0, out, sstore, sout, ds0,
                               L=L, G=G, KV=kv, H=H, DH=DH, VH=VH, VSPLIT=VSPLIT, KVG=kvg, STAGES=STAGES, HAS_ROT=HAS_ROT_, HAS_S0=HAS_S0_, RETURN_STATE=return_state, HAS_DS0=needs)
        ctx.save_for_backward(k, v, rate, eta, q, grot, sstore, sout, ds0)
        ctx.kv, ctx.kvg = kv, kvg
        ctx.return_state = return_state
        if return_state:
            return out, sout
        return out""", "F2")
rep("    def backward(ctx, dy):", "    def backward(ctx, dy, ds_last=None):", "B1s")

# 4) Function backward: 传 DSL/DS0 + 返回 dsl/ds0
rep("""        HAS_S0_ = 1 if ctx.has_s0 else 0
        grid = (B * G * VSPLIT,)
        delta_bwd_kernel[grid](k, v, rate, eta, q, grot, dy, sstore, dk, dv, dr, de, dq, dg,
                               L=L, G=G, KV=ctx.kv, H=H, DH=DH, VH=VH, VSPLIT=VSPLIT, KVG=ctx.kvg, STAGES=STAGES, HAS_ROT=HAS_ROT_, HAS_S0=HAS_S0_)
        return (dk.to(k.dtype), dv.to(v.dtype), dr.to(rate.dtype), de.to(eta.dtype),
                dq.to(q.dtype), (dg.to(grot.dtype) if ctx.has_rot else None),
                None, None, None, None)""",
    """        HAS_S0_ = 1 if ctx.has_s0 else 0
        HAS_DSL_ = 1 if ds_last is not None else 0
        dsl_t = ds_last.contiguous().float() if ds_last is not None else dv
        ds0_buf = torch.zeros(B, G, DH, DH, device=k.device, dtype=torch.float32)
        grid = (B * G * VSPLIT,)
        delta_bwd_kernel[grid](k, v, rate, eta, q, grot, dy, sstore, dk, dv, dr, de, dq, dg, dsl_t, ds0_buf,
                               L=L, G=G, KV=ctx.kv, H=H, DH=DH, VH=VH, VSPLIT=VSPLIT, KVG=ctx.kvg, STAGES=STAGES, HAS_ROT=HAS_ROT_, HAS_S0=HAS_S0_, HAS_DSL=HAS_DSL_, HAS_DS0=HAS_S0_)
        dsl_out = ds_last.to(k.dtype) if ds_last is not None else None
        return (dk.to(k.dtype), dv.to(v.dtype), dr.to(rate.dtype), de.to(eta.dtype),
                dq.to(q.dtype), (dg.to(grot.dtype) if ctx.has_rot else None),
                (ds0_buf.to(k.dtype) if ctx.has_s0 else None),
                dsl_out, None, None, None, (dsl_out if ctx.return_state else None))""", "B3")

# 5) wrapper
rep("def delta_scan_triton(k, v, rate, eta, q, kv, slots, kvg, grot=None, s0=None):",
    "def delta_scan_triton(k, v, rate, eta, q, kv, slots, kvg, grot=None, s0=None, return_state=False):", "W1")
rep("    return DeltaScanTriton.apply(k, v, rate, eta, q, grot, s0, kv, slots, kvg)",
    "    return DeltaScanTriton.apply(k, v, rate, eta, q, grot, s0, kv, slots, kvg, return_state)", "W2")

io.open(p, "w", encoding="utf-8").write(t)
print("miss:", miss if miss else "无 — 全部应用")
