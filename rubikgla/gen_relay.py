# -*- coding: utf-8 -*-
"""从 v31_spd 生成 deqlm_v35_relay.py: s0 接力注入 + 末态捕获."""
import io

src = io.open(r"F:\夸克\rubikgla\deqlm_v31_spd.py", encoding="utf-8").read()

# 1) _scan_core 加 s0
src = src.replace(
    "    def _scan_core(self, x, itmult):",
    "    def _scan_core(self, x, itmult, s0=None):\n"
    "        if self.capture:\n"
    "            import delta_triton as _dt\n"
    "            B, L, _ = x.shape\n"
    "            G = self.KV * self.S\n"
    "            hh = self.ln(x)\n"
    "            qq = self.wq(hh).view(B, L, H, self.dh)\n"
    "            kk = self.wk(hh).view(B, L, G, self.dh)\n"
    "            vv = self.wv(hh).view(B, L, G, self.dh)\n"
    "            vf = vv.reshape(B, L, -1).transpose(1, 2)\n"
    "            import torch.nn.functional as _F\n"
    "            vc = _F.conv1d(_F.pad(vf, (3, 0)), self.dwconv.weight, groups=G * self.dh)\n"
    "            vv = (vf + vc).transpose(1, 2).reshape(B, L, G, self.dh)\n"
    "            rr = _F.softplus(self.w_rate(hh)).view(B, L, G, self.dh) * itmult\n"
    "            ee = torch.sigmoid(self.w_eta(hh)).view(B, L, G, self.dh)\n"
    "            self._last_state, yv = _dt.delta_scan_states(kk, vv, rr, ee, qq, KV_H, SLOTS, H // KV_H, s0=s0)\n"
    "            return yv.view(B, L, KV_H, SLOTS, H // KV_H, DH).sum(3).reshape(B, L, H * DH)",
    1)
src = src.replace(
    "        return DELTA_FN(k, v, rate, eta, q)",
    "        return DELTA_FN(k, v, rate, eta, q, s0)", 1)

# 2) _apply_raw / forward 加 s0
src = src.replace("    def _apply_raw(self, x, itmult):", "    def _apply_raw(self, x, itmult, s0=None):", 1)
src = src.replace("        o = self._scan_core(x, itmult)", "        o = self._scan_core(x, itmult, s0)", 1)
src = src.replace("    def forward(self, x, itmult):", "    def forward(self, x, itmult, s0=None):", 1)
src = src.replace("            out = checkpoint(self._scan_core, x, itmult, use_reentrant=False)",
                  "            out = checkpoint(self._scan_core, x, itmult, s0, use_reentrant=False)", 1)
src = src.replace("                out = self._scan_core(x, itmult)",
                  "                out = self._scan_core(x, itmult, s0)", 1)

# 3) DeltaBlock.forward 加 s0
src = src.replace("    def forward(self, x, it=0):\n        mult = self._mult(it)",
                  "    def forward(self, x, it=0, s0=None):\n        mult = self._mult(it)", 1)
src = src.replace("            o = torch.utils.checkpoint.checkpoint(self._attn_impl, x, mult, use_reentrant=False)",
                  "            o = torch.utils.checkpoint.checkpoint(self._attn_impl, x, mult, s0, use_reentrant=False)", 1)
src = src.replace("            return self._ffn_impl(o)", "            return self._ffn_impl(o)", 1)
src = src.replace("        with torch.no_grad():\n            return self.ffn(self.attn._apply_raw(x, mult))",
                  "        with torch.no_grad():\n            return self.ffn(self.attn._apply_raw(x, mult, s0))", 1)

# 4) TiedDeltaBlock 加 s0 (仅 j==0 注入)
src = src.replace("    def forward(self, z0):\n        alpha = self.alpha()\n        z = z0\n        for j in range(self.kmax):\n            z = z + alpha * self.attn(z, it=j)\n        return z",
                  "    def forward(self, z0, s0=None):\n        alpha = self.alpha()\n        z = z0\n        for j in range(self.kmax):\n            z = z + alpha * self.attn(z, it=j, s0=(s0 if j == 0 else None))\n        return z", 1)

# 5) DeltaLoopLM.forward 加 s0_list + capture_states
src = src.replace("    def forward(self, ids):", "    def forward(self, ids, s0_list=None):", 1)
src = src.replace(
    "        h = self.emb(ids)\n        h = self.pre(h)\n        h = self.loop_a(h)\n        h = self.bridge(h)\n        h = self.loop_c(h)\n        h = self.post(h)\n        return self.head(self.ln_f(h))",
    "        s0s = s0_list if s0_list is not None else [None] * 5\n"
    "        h = self.emb(ids)\n"
    "        h = self.pre(h, s0=s0s[0])\n"
    "        h = self.loop_a(h, s0=s0s[1])\n"
    "        h = self.bridge(h, s0=s0s[2])\n"
    "        h = self.loop_c(h, s0=s0s[3])\n"
    "        h = self.post(h, s0=s0s[4])\n"
    "        return self.head(self.ln_f(h))\n\n"
    "    def capture_states(self, ids):\n"
    "        \"\"\"无梯度前向, 返回 5 站末状态 [(B,G,DH,DH)] (接力接口).\"\"\"\n"
    "        was = [blk.attn.capture for blk in (self.pre, self.loop_a.attn, self.bridge, self.loop_c.attn, self.post)]\n"
    "        for blk in (self.pre, self.loop_a.attn, self.bridge, self.loop_c.attn, self.post):\n"
    "            blk.attn.capture = True\n"
    "            blk.attn._last_state = None\n"
    "        self.eval()\n"
    "        with torch.no_grad():\n"
    "            self.forward(ids)\n"
    "        states = [blk.attn._last_state for blk in (self.pre, self.loop_a.attn, self.bridge, self.loop_c.attn, self.post)]\n"
    "        for blk, w in zip((self.pre, self.loop_a.attn, self.bridge, self.loop_c.attn, self.post), was):\n"
    "            blk.attn.capture = w\n"
    "        return states", 1)

# 6) DeltaAttn capture 标志
src = src.replace("        self.rho_p = nn.Parameter(torch.full((KV_H, SLOTS, self.dh), math.log(0.8 / 0.2)))",
                  "        self.rho_p = nn.Parameter(torch.full((KV_H, SLOTS, self.dh), math.log(0.8 / 0.2)))\n"
                  "        self.capture = False\n"
                  "        self._last_state = None", 1)

io.open(r"F:\夸克\rubikgla\deqlm_v35_relay.py", "w", encoding="utf-8").write(src)
print("wrote deqlm_v35_relay.py")
