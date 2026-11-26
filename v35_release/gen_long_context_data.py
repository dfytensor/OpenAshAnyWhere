"""
生成长上下文 SFT 数据集
包含 3 类任务:
1. Needle-in-Haystack: 在长文本中插入事实，末尾提问
2. Long-Distance Copy: 远距离复制信息
3. Multi-turn memory: 多轮对话中引用早期信息
"""
import json, random, os

random.seed(42)

# 小说文本作为 haystack
novel_dir = r"F:\小说\女生小说"
novel_texts = []
for f in os.listdir(novel_dir):
    if f.endswith('.txt'):
        try:
            with open(os.path.join(novel_dir, f), 'r', encoding='utf-8', errors='ignore') as fp:
                text = fp.read(500000)
            if len(text) > 10000:
                novel_texts.append(text)
            if len(novel_texts) >= 20:
                break
        except:
            continue

print(f"Loaded {len(novel_texts)} novels as haystack")

# 事实池
NEEDLES = [
    ("我的手机密码是8473", "我的手机密码是什么", "8473"),
    ("张三的银行卡号是6225880137123456", "张三的银行卡号后四位是多少", "3456"),
    ("钥匙藏在门口花盆下面第三个位置", "钥匙藏在哪里", "花盆"),
    ("会议室在三楼302房间", "会议室在哪个房间", "302"),
    ("这本书的作者是李华", "这本书的作者是谁", "李华"),
    ("密码箱的密码是9527", "密码箱的密码是多少", "9527"),
    ("小明家的猫叫橘子", "小明家的猫叫什么", "橘子"),
    ("今天是2026年6月14日", "今天是几月几号", "6月14日"),
    ("火车票的价格是350元", "火车票多少钱", "350"),
    ("项目的截止日期是12月31日", "项目截止日期是什么时候", "12月31日"),
    ("老师的电话号码是13800138000", "老师的电话号码是多少", "13800138000"),
    ("快递放在了小区门口的丰巢柜里", "快递放在哪里", "丰巢柜"),
    ("王医生的诊室在五楼508", "王医生在哪个诊室", "508"),
    ("考试的总分是750分", "考试总分是多少", "750"),
    ("餐厅的预订时间是晚上七点", "餐厅预订了几点", "七点"),
    ("李经理的工号是A20250314", "李经理的工号是什么", "A20250314"),
    ("仓库的入口在北侧B区", "仓库入口在哪里", "B区"),
    ("合同编号是HT-2026-0518", "合同编号是什么", "HT-2026-0518"),
    ("样品的温度需要保持在零下20度", "样品需要什么温度", "零下20度"),
    ("飞机航班号是CA1985", "航班号是多少", "CA1985"),
]

def get_haystack_chunk(target_tokens=300):
    """从随机小说中截取一段文本"""
    novel = random.choice(novel_texts)
    start = random.randint(0, max(0, len(novel) - target_tokens * 3))
    chunk = novel[start:start + target_tokens * 3]
    return chunk

def make_needle_sample(max_needles=3):
    """生成大海捞针样本: haystack中插入1-3个针，末尾提问"""
    n_needles = random.randint(1, max_needles)
    chosen = random.sample(NEEDLES, n_needles)
    
    haystack = get_haystack_chunk(random.randint(200, 500))
    # 在haystack中不同位置插入针
    positions = sorted(random.sample(range(len(haystack) // 5, len(haystack) * 4 // 5), n_needles))
    
    insert_text = haystack
    offset = 0
    for i, pos in enumerate(positions):
        stmt = chosen[i][0]
        actual_pos = pos + offset
        insert_text = insert_text[:actual_pos] + stmt + "。" + insert_text[actual_pos:]
        offset += len(stmt) + 1
    
    # 随机问一个针
    ask_idx = random.randint(0, n_needles - 1)
    _, question, answer = chosen[ask_idx]
    
    conv = [
        {"role": "user", "content": insert_text + "\n\n" + question},
        {"role": "assistant", "content": answer},
    ]
    return {"conversations": conv}

def make_copy_sample():
    """远距离复制: 给一个标记词，中间大量文本，末尾问标记词"""
    markers = ["凤凰", "星辰", "海洋", "山巅", "月光", "黎明", "翡翠", "琥珀", "珊瑚", "玛瑙"]
    marker = random.choice(markers)
    
    haystack = get_haystack_chunk(random.randint(200, 600))
    
    conv = [
        {"role": "user", "content": f"请记住这个关键词：{marker}。\n\n{haystack}\n\n刚才让你记住的关键词是什么？"},
        {"role": "assistant", "content": marker},
    ]
    return {"conversations": conv}

def make_summary_sample():
    """长文本理解: 给一段长文本，要求总结开头提到的人名/地名"""
    haystack = get_haystack_chunk(random.randint(300, 500))
    
    # 在开头插入关键信息
    names = ["李明", "王芳", "张伟", "刘洋", "陈静", "赵雷", "孙莉", "周杰"]
    places = ["北京", "上海", "广州", "深圳", "杭州", "成都", "西安", "南京"]
    name = random.choice(names)
    place = random.choice(places)
    
    intro = f"{name}在{place}工作。"
    full_text = intro + haystack
    
    questions = [
        (f"上文提到的人在哪个城市工作？", f"{place}"),
        (f"上文提到的第一个人叫什么名字？", f"{name}"),
        (f"文中提到的人名是什么？", f"{name}"),
    ]
    q, a = random.choice(questions)
    
    conv = [
        {"role": "user", "content": full_text + "\n\n" + q},
        {"role": "assistant", "content": a},
    ]
    return {"conversations": conv}

def make_multiturn_sample():
    """多轮记忆: 前几轮提供信息，后面提问"""
    n_turns = random.randint(3, 6)
    
    info_needles = random.sample(NEEDLES, min(3, n_turns // 2))
    conv = []
    
    # 前几轮: 提供信息
    for i, (stmt, _, _) in enumerate(info_needles):
        conv.append({"role": "user", "content": f"请记住：{stmt}"})
        conv.append({"role": "assistant", "content": "好的，我记住了。"})
    
    # 中间填充闲聊
    fillers = [
        ("今天天气怎么样？", "我无法感知天气，但希望您有个愉快的一天。"),
        ("你喜欢什么颜色？", "作为AI我没有偏好，但我觉得蓝色很优雅。"),
        ("1+1等于几？", "1+1等于2。"),
        ("最近有什么新闻？", "我无法获取实时信息，建议您查看新闻网站。"),
    ]
    for q, a in random.sample(fillers, min(2, n_turns - len(info_needles))):
        conv.append({"role": "user", "content": q})
        conv.append({"role": "assistant", "content": a})
    
    # 最后一轮: 提问之前的信息
    ask = random.choice(info_needles)
    conv.append({"role": "user", "content": ask[1]})
    conv.append({"role": "assistant", "content": ask[2]})
    
    return {"conversations": conv}

# 生成数据集
OUTPUT = "minimind_data/long_context_sft.jsonl"
TOTAL = 5000

print(f"Generating {TOTAL} long-context SFT samples...")

with open(OUTPUT, 'w', encoding='utf-8') as f:
    for i in range(TOTAL):
        r = random.random()
        if r < 0.4:
            sample = make_needle_sample(max_needles=random.choice([1, 2, 3]))
        elif r < 0.6:
            sample = make_copy_sample()
        elif r < 0.8:
            sample = make_summary_sample()
        else:
            sample = make_multiturn_sample()
        
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"Done. Written to {OUTPUT}")

# 统计
lengths = []
with open(OUTPUT, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        total_len = sum(len(m['content']) for m in item['conversations'])
        lengths.append(total_len)

print(f"Avg content length: {sum(lengths)/len(lengths):.0f} chars")
print(f"Max: {max(lengths)}, Min: {min(lengths)}")
print(f">500 chars: {sum(1 for l in lengths if l > 500)}/{len(lengths)}")
print(f">1000 chars: {sum(1 for l in lengths if l > 1000)}/{len(lengths)}")
