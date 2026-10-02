import ast

p = r"F:\夸克\rubikgla\deqlm_v22_hybrid2.py"
c = open(p, encoding="utf-8").read()
old1 = 'alphas = " ".join("%.3f" % lp.alpha().item() for lp in m.loops)'
new1 = 'alphas = "%.3f/%.3f" % (m.loop_a.alpha().item(), m.loop_c.alpha().item())'
old2 = 'alphas=[round(lp.alpha().item(), 4) for lp in m.loops]'
new2 = 'alphas=[round(m.loop_a.alpha().item(), 4), round(m.loop_c.alpha().item(), 4)]'
assert old1 in c, "old1 missing"
assert old2 in c, "old2 missing"
c = c.replace(old1, new1).replace(old2, new2)
open(p, "w", encoding="utf-8").write(c)
ast.parse(c)
print("patched + syntax OK")
