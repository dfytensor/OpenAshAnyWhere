import io
p = r"F:\夸克\rubikgla\relay_joint.py"
t = io.open(p, encoding="utf-8").read()
t = t.replace("lr = 4e-4 * min(1.0, st / 1000)", "lr = 4e-4 * min(1.0, st / 2000)")
t = t.replace("h_proj.unsqueeze(1)", "0.1 * h_proj.unsqueeze(1)")
t = t.replace(
    "log(\"%d/%d loss=%.4f gn=%.2f (%.0fms)\"",
    "log(\"%d/%d loss=%.4f gn=%.2f lr=%.2e (%.0fms)\"")
t = t.replace(
    "% (st, STEPS, loss.item(), gn.item(), (time.time()-t0)/(st+1)*1000))",
    "% (st, STEPS, loss.item(), gn.item(), lr, (time.time()-t0)/(st+1)*1000))")
io.open(p, "w", encoding="utf-8").write(t)
print("patched: lr warmup 2000 + injection 0.1 + log lr")
