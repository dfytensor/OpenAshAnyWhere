import ast

src = r"F:\夸克\rubikgla\deqlm_v20_tunedloop.py"
dst = r"F:\夸克\rubikgla\deqlm_v20ctrl_12k.py"
c = open(src, encoding="utf-8").read()
c = c.replace("PT_STEPS = 39695", "PT_STEPS = 12000")
c = c.replace("deqlm20.log", "deqlm20c.log")
c = c.replace("deqlm20_results.json", "deqlm20c_results.json")
c = c.replace("deqlm20_pt_full.pth", "deqlm20c_pt_full.pth")
c = c.replace('log("v20: %.2fM', 'log("v20ctrl(12k): %.2fM')
open(dst, "w", encoding="utf-8").write(c)
ast.parse(c)
print("control script written:", dst)
