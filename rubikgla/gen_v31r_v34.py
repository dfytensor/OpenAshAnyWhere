# -*- coding: utf-8 -*-
"""生成 v31r (B32+哨兵) 与 v34 (加深13) 训练脚本."""
import io

src = io.open(r"F:\夸克\rubikgla\deqlm_v31_spd.py", encoding="utf-8").read()

old_loop = """            x, y = get_batch(train, bs=B_MICRO, device=DEV, rng=rng)
            x2, y2 = get_batch(train, bs=B_MICRO, device=DEV, rng=rng)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = m(x)
            l1 = F.cross_entropy(lo.float().reshape(-1, V), y.reshape(-1), ignore_index=0)
            if torch.isfinite(l1):
                (l1 * 0.5).backward()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo2 = m(x2)
            l2 = F.cross_entropy(lo2.float().reshape(-1, V), y2.reshape(-1), ignore_index=0)
            if torch.isfinite(l2):
                (l2 * 0.5).backward()
            if not (torch.isfinite(l1) and torch.isfinite(l2)):
                opt.zero_grad(set_to_none=True)
                continue
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if st % 2000 == 0:
                alphas = "%.3f/%.3f" % (m.loop_a.alpha().item(), m.loop_c.alpha().item())
                log("%d/%d loss=%.4f a=[%s] (%.0fms/步, ETA %.1f时)"
                    % (st, PT_STEPS, (0.5 * (l1 + l2)).item(), alphas,
                       (time.time() - t0) / (st - start_step + 1) * 1000,
                       (PT_STEPS - st) * (time.time() - t0) / (st - start_step + 1) / 3600))"""

new_loop = """            x, y = get_batch(train, bs=B_MICRO, device=DEV, rng=rng)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = m(x)
            l1 = F.cross_entropy(lo.float().reshape(-1, V), y.reshape(-1), ignore_index=0)
            if not torch.isfinite(l1):
                opt.zero_grad(set_to_none=True)
                nan_run += 1
                if nan_run > 200:
                    log("%(TAG)s NaN 失控 @step %d — 存断点退出" % st)
                    torch.save(dict(model=m.state_dict(), opt=opt.state_dict(),
                                    rng=rng.getstate(), st=st), RESUME)
                    raise RuntimeError("NaN runaway at step %d" % st)
                continue
            nan_run = 0
            (l1).backward()
            gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            if st %% 2000 == 0:
                alphas = "%%.3f/%%.3f" %% (m.loop_a.alpha().item(), m.loop_c.alpha().item())
                log("%%d/%%d loss=%%.4f a=[%%s] (%%.0fms/步, ETA %%.1f时)"
                    %% (st, PT_STEPS, l1.item(), alphas,
                       (time.time() - t0) / (st - start_step + 1) * 1000,
                       (PT_STEPS - st) * (time.time() - t0) / (st - start_step + 1) / 3600))"""

assert old_loop in src, "loop block not found"
src = src.replace(old_loop, new_loop.replace("%(TAG)s", "{TAG}").replace("%%", "%"))
src = src.replace("B_MICRO = 16", "B_MICRO = 32")
src = src.replace("    t0 = time.time()\n    B_MICRO = 32", "    t0 = time.time()\n    B_MICRO = 32\n    nan_run = 0")


def emit(tag, ks=None):
    t = src.format(TAG=tag)
    t = t.replace("deqlm31s.log", "deqlm%s.log" % tag)
    t = t.replace("deqlm31s_results.json", "deqlm%s_results.json" % tag)
    t = t.replace("deqlm31s_pt_full.pth", "deqlm%s_pt_full.pth" % tag)
    t = t.replace("deqlm31s_resume.pth", "deqlm%s_resume.pth" % tag)
    t = t.replace('log("v31s:', 'log("%s:' % tag.upper())
    t = t.replace("v31s 续跑", "%s 续跑" % tag)
    t = t.replace("v31s 异常", "%s 异常" % tag)
    t = t.replace("v31s 完成", "%s 完成" % tag)
    if ks:
        t = t.replace("KS = [10, 10, 10]", "KS = [%d, 10, %d]" % (ks, ks))
    fn = r"F:\夸克\rubikgla\deqlm_%s.py" % ("v31r_b32" if tag == "v31r" else "v34_deep13")
    io.open(fn, "w", encoding="utf-8").write(t)
    print("wrote", fn)


emit("v31r")
emit("v34", ks=13)
print("DONE")
