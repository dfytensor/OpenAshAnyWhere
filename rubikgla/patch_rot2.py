# -*- coding: utf-8 -*-
"""对已提交的 delta_triton.py(S0 版) 加: return_state + dS0 输出 + ds_last 注入."""
import io
p = r"F:\夸克\rubikgla\delta_triton.py"
t = io.open(p, encoding="utf-8").read()
miss = []
def rep(old, new, tag):
    global t
    if old not in t:
        miss.append(tag); return
    t = t.replace(old, new, 1)

# fwd: SOUT + RETURN_STATE
rep("def delta_fwd_kernel(\n    K, V, RATE, ETA, Q, GROT, S0, OUT, SSTORE,",
    "def delta_fwd_kernel(\n    K, V, RATE, ETA, Q, GROT, S0, OUT, SSTORE, SOUT,", "f1")
rep("HAS_S0: tl.constexpr,\n):", "HAS_S0: tl.constexpr, RETURN_STATE: tl.constexpr,\n):", "f2")
rep("""        for m in tl.static_range(KVG):
            qm = tl.load(q_base + t * H * DH + m * DH + offs_i).to(tl.float32)
            ym = tl.sum(qm[:, None] * S, axis=0)
            tl.store(o_base + t * G * KVG * DH + m * DH + offs_js,
                     ym.to(OUT.dtype.element_ty))""",
    """        for m in tl.static_range(KVG):
            qm = tl.load(q_base + t * H * DH + m * DH + offs_i).to(tl.float32)
            ym = tl.sum(qm[:, None] * S, axis=0)
            tl.store(o_base + t * G * KVG * DH + m * DH + offs_js,
                     ym.to(OUT.dtype.element_ty))
    if RETURN_STATE:
        so_base = SOUT + (b * G + g) * DH * DH
        tl.store(so_base + offs_i[:, None] * DH + offs_js[None, :], S.to(SOUT.dtype.element_ty))""", "f3")

# bwd: DSL + DS0 参数与写点
rep("def delta_bwd_kernel(\n    K, V, RATE, ETA, Q, GROT, DY, SSTORE, DK, DV, DRATE, DETA, DQ, DGROT,",
    "def delta_bwd_kernel(\n    K, V, RATE, ETA, Q, GROT, DY, SSTORE, DK, DV, DRATE, DETA, DQ, DGROT, DSL, DS0,", "b1")
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

# Function
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
    """        out = torch.empty(B, L, G, kvg, DH, device=k.device, dtype=torch.bfloat16)
        sstore = torch.empty(B, L, G, DH, DH, device=k.device, dtype=torch.float32)
        needs = ctx.needs_s0_grad if hasattr(ctx, 'needs_s0_grad') else False
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
rep("""    def forward(ctx, k, v, rate, eta, q, grot, s0, kv, slots, kvg, return_state=False):
        B, L, G, DH = k.shape""",
    """    def forward(ctx, k, v, rate, eta, q, grot, s0, kv, slots, kvg, return_state=False):
        B, L, G, DH = k.shape
        needs = s0 is not None and s0.requires_grad
        ctx.needs_s0_grad = needs""", "F2b")
rep("        ctx.save_for_backward(k, v, rate, eta, q, grot, sstore)",
    "        ctx.save_for_backward(k, v, rate, eta, q, grot, sstore, ds0)", "F2c")
rep("    def backward(ctx, dy):", "    def backward(ctx, dy, ds_last=None):", "B1")
rep("""        ctx.saved_tensors""", """        (k, v, rate, eta, q, grot, sstore, sout, ds0_buf) = ctx.saved_tensors""", "B2")
rep("""        HAS_S0_ = 1 if ctx.has_s0 else 0
        grid = (B * G * VSPLIT,)
        delta_bwd_kernel[grid](k, v, rate, eta, q, grot, dy, sstore, dk, dv, dr, de, dq, dg,
                               L=L, G=G, KV=ctx.kv, H=H, DH=DH, VH=VH, VSPLIT=VSPLIT, KVG=ctx.kvg, STAGES=STAGES, HAS_ROT=HAS_ROT_, HAS_S0=HAS_S0_)
        return (dk.to(k.dtype), dv.to(v.dtype), dr.to(rate.dtype), de.to(eta.dtype),
                dq.to(q.dtype), (dg.to(grot.dtype) if ctx.has_rot else None),
                None, None, None, None)""",
    """        HAS_S0_ = 1 if ctx.has_s0 else 0
        use_ds0 = ds0_buf.numel() > 1
        grid = (B * G * VSPLIT,)
        if use_ds0:
            delta_bwd_kernel[grid](k, v, rate, eta, q, grot, dy, sstore, dk, dv, dr, de, dq, dg, ds0_buf,
                                   L=L, G=G, KV=ctx.kv, H=H, DH=DH, VH=VH, VSPLIT=VSPLIT, KVG=ctx.kvg, STAGES=STAGES, HAS_ROT=HAS_ROT_, HAS_S0=HAS_S0_, HAS_DSL=ds_last is not None, HAS_DS0=1, DSL=ds_last if ds_last is not None else dv)
        else:
            delta_bwd_kernel[grid](k, v, rate, eta, q, grot, dy, sstore, dk, dv, dr, de, dq, dg, dv,
                                   L=L, G=G, KV=ctx.kv, H=H, DH=DH, VH=VH, VSPLIT=VSPLIT, KVG=ctx.kvg, STAGES=STAGES, HAS_ROT=HAS_ROT_, HAS_DSL=ds_last is not None, HAS_DS0=0, DSL=dv if ds_last is not None else dv)
        dsl_out = ds_last.to(k.dtype) if ds_last is not None else None
        return (dk.to(k.dtype), dv.to(v.dtype), dr.to(rate.dtype), de.to(eta.dtype),
                dq.to(q.dtype), (dg.to(grot.dtype) if ctx.has_rot else None),
                (ds0_buf.to(k.dtype) if use_ds0 else None),
                dsl_out, None, None, None, (dsl_out if ctx.return_state else None))""", "B3")

# wrapper
rep("def delta_scan_triton(k, v, rate, eta, q, kv, slots, kvg, grot=None, s0=None):",
    "def delta_scan_triton(k, v, rate, eta, q, kv, slots, kvg, grot=None, s0=None, return_state=False):", "W1")
rep("    return DeltaScanTriton.apply(k, v, rate, eta, q, grot, s0, kv, slots, kvg)",
    "    return DeltaScanTriton.apply(k, v, rate, eta, q, grot, s0, kv, slots, kvg, return_state)", "W2")

io.open(p, "w", encoding="utf-8").write(t)
print("miss:", miss if miss else "无")
