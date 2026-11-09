"""批量下载 C-Eval 各科目 dev.csv（hf-mirror 镜像）。"""
import os
import sys
import urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SUBJECTS = ['accountant', 'advanced_mathematics', 'art_studies', 'basic_medicine',
            'business_administration', 'chinese_language_and_literature', 'civil_servant',
            'clinical_medicine', 'college_chemistry', 'college_economics', 'college_physics',
            'college_programming', 'computer_architecture', 'computer_network',
            'discrete_mathematics', 'education_science', 'electrical_engineer',
            'environmental_impact_assessment_engineer', 'fire_engineer', 'high_school_biology',
            'high_school_chemistry', 'high_school_chinese', 'high_school_geography',
            'high_school_history', 'high_school_mathematics', 'high_school_physics',
            'high_school_politics', 'ideological_and_moral_cultivation', 'law',
            'legal_professional', 'logic', 'marxism', 'metrology', 'middle_school_biology',
            'middle_school_chemistry', 'middle_school_geography', 'middle_school_history',
            'middle_school_mathematics', 'middle_school_physics', 'middle_school_politics',
            'modern_chinese_history', 'operating_system', 'physician', 'plant_protection',
            'probability_and_statistics', 'professional_accountant', 'registered_architect',
            'structural_engineer', 'tax_accountant', 'teacher_qualification', 'urban_and_rural_planner',
            'veterinary_medicine']

OUT = r'F:\OpenASH2605\ceval_data'
os.makedirs(OUT, exist_ok=True)
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}


def fetch(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


# 试第一个科目的 dev.csv 确认格式
u0 = f'https://hf-mirror.com/datasets/ceval/ceval-exam/resolve/main/accountant/dev.csv'
try:
    d = fetch(u0)
    print('accountant/dev.csv:', len(d), 'bytes | head:', d[:150].decode('utf-8', 'replace'))
except Exception as e:
    print('accountant dev fail:', repr(e)[:120])
    # 试 val.csv 作为替代（hf 版可能带答案）
    try:
        d = fetch(f'https://hf-mirror.com/datasets/ceval/ceval-exam/resolve/main/accountant/val.csv')
        print('accountant/val.csv:', len(d), 'bytes | head:', d[:150].decode('utf-8', 'replace'))
    except Exception as e2:
        print('val fail:', repr(e2)[:120])
    sys.exit(1)

ok, fail = 0, []
for s in SUBJECTS:
    try:
        d = fetch(f'https://hf-mirror.com/datasets/ceval/ceval-exam/resolve/main/{s}/dev.csv')
        with open(os.path.join(OUT, f'{s}_dev.csv'), 'wb') as f:
            f.write(d)
        ok += 1
    except Exception as e:
        fail.append(s)
print(f'downloaded {ok}/{len(SUBJECTS)} | failed: {fail[:6]}')
