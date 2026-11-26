#!/usr/bin/env python3
import os
import sys
import io
import json
import re
import random
import datetime

import streamlit as st
import torch
import numpy as np
from threading import Thread

from open_ash import OpenASH
from open_ash_voc import OpenASHVoc
from config import agent_voc_path

_orig_stdout = sys.stdout
_orig_stderr = sys.stderr


class _CompatStdout:
    def __init__(self, stream):
        self._stream = stream
        self.buffer = io.BytesIO()

    def write(self, *a, **kw):
        try:
            return self._stream.write(*a, **kw)
        except (ValueError, AttributeError):
            return 0

    def flush(self, *a, **kw):
        try:
            return self._stream.flush(*a, **kw)
        except (ValueError, AttributeError):
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


sys.stdout = _CompatStdout(sys.stdout)
sys.stderr = _CompatStdout(sys.stderr)

from open_ash_infer import (
    build_chat_prompt, generate_stream, format_response,
    sample_next_token, _sp, MAX_SEQ_LEN
)

sys.stdout = _orig_stdout
sys.stderr = _orig_stderr

st.set_page_config(page_title="OpenASH", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        .stButton button {
            border-radius: 50% !important;
            width: 32px !important;
            height: 32px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;
            margin: 5px 10px 5px 0 !important;
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        .stMainBlockContainer > div:first-child {
            margin-top: -50px !important;
        }
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }
        .stButton > button {
            all: unset !important;
            box-sizing: border-box !important;
            border-radius: 50% !important;
            width: 18px !important;
            height: 18px !important;
            min-width: 18px !important;
            min-height: 18px !important;
            max-width: 18px !important;
            max-height: 18px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #888 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            margin: 0 2px !important;
        }
    </style>
""", unsafe_allow_html=True)

WEIGHT_DIR = r"F:\OpenASH\out"
HIDDEN_SIZE = 640
NUM_LAYERS = 8
NUM_HEADS = 8
MAX_SEQ_LEN = 2048

LANG_TEXTS = {
    'zh': {
        'settings': '模型设定',
        'history_rounds': '历史对话轮次',
        'max_length': '最大生成长度',
        'temperature': '温度',
        'top_k': 'Top-K',
        'top_p': 'Top-P',
        'repetition_penalty': '重复惩罚',
        'thinking': '思考',
        'tools': '工具',
        'language': '语言',
        'send': '给 OpenASH 发送消息',
        'disclaimer': 'AI 生成内容可能存在错误，请仔细核实',
        'think_tip': '开启思考模式',
        'tool_select': '工具选择（最多4个）',
    },
    'en': {
        'settings': 'Model Settings',
        'history_rounds': 'History Rounds',
        'max_length': 'Max Length',
        'temperature': 'Temperature',
        'top_k': 'Top-K',
        'top_p': 'Top-P',
        'repetition_penalty': 'Repetition Penalty',
        'thinking': 'Thinking',
        'tools': 'Tools',
        'language': 'Language',
        'send': 'Send a message to OpenASH',
        'disclaimer': 'AI-generated content may be inaccurate, please verify',
        'think_tip': 'Enable thinking mode',
        'tool_select': 'Tool Selection (max 4)',
    }
}

def get_text(key):
    lang = st.session_state.get('lang', 'en')
    return LANG_TEXTS.get(lang, {}).get(key, LANG_TEXTS['zh'].get(key, key))


TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}, "required": ["min", "max"]}}},
    {"type": "function", "function": {"name": "text_length", "description": "计算文本长度", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位转换", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "获取汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}}, "required": ["text", "target_lang"]}}},
]

TOOL_SHORT_NAMES = {
    'calculate_math': '数学', 'get_current_time': '时间', 'random_number': '随机',
    'text_length': '字数', 'unit_converter': '单位', 'get_current_weather': '天气',
    'get_exchange_rate': '汇率', 'translate_text': '翻译'
}


def execute_tool(tool_name, args):
    try:
        if tool_name == 'calculate_math':
            return {"result": str(eval(args.get('expression', '0')))}
        elif tool_name == 'get_current_time':
            return {"result": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        elif tool_name == 'random_number':
            return {"result": random.randint(args.get('min', 0), args.get('max', 100))}
        elif tool_name == 'text_length':
            return {"result": len(args.get('text', ''))}
        elif tool_name == 'unit_converter':
            return {"result": f"{args.get('value', 0)} {args.get('from_unit', '')} = ? {args.get('to_unit', '')}"}
        elif tool_name == 'get_current_weather':
            return {"result": f"{args.get('city', 'Unknown')}: 晴, 7~10°C"}
        elif tool_name == 'get_exchange_rate':
            return {"result": f"1 {args.get('from_currency', 'USD')} = 7.2 {args.get('to_currency', 'CNY')}"}
        elif tool_name == 'translate_text':
            return {"result": f"[翻译结果]: hello world"}
        return {"result": "Unknown tool"}
    except Exception as e:
        return {"error": str(e)}


def format_thinking_html(think_text, finished=True):
    label = "已思考" if finished else "思考中..."
    style_scroll = "" if finished else "display: flex; flex-direction: column-reverse;"
    return (
        f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;">'
        f'<summary style="cursor: pointer; color: #888;">{label}</summary>'
        f'<div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto; {style_scroll}">'
        f'<div style="margin-bottom: auto;">{think_text.strip()}</div></div></details>'
    )


def format_tool_call_html(name, args_str):
    return (
        f'<div style="background: rgba(80,110,150,0.20); border: 1px solid rgba(140,170,210,0.30); '
        f'padding: 10px 12px; border-radius: 12px; margin: 6px 0;">'
        f'<div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolCalling</div>'
        f'<div><b>{name}</b>: {args_str}</div></div>'
    )


def format_tool_result_html(name, result_str):
    return (
        f'<div style="background: rgba(90,130,110,0.20); border: 1px solid rgba(150,200,170,0.30); '
        f'padding: 10px 12px; border-radius: 12px; margin: 6px 0;">'
        f'<div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolResult</div>'
        f'<div><b>{name}</b>: {result_str}</div></div>'
    )


_SPECIAL_TOKEN_RE = re.compile(r'</?\|?[a-z_]+\|?>')


def _strip_special(text):
    text = _SPECIAL_TOKEN_RE.sub('', text)
    text = text.replace('ALSE', '').replace('ALND', '')
    return text.strip()


def process_assistant_content(raw_text, is_streaming=False):
    content = raw_text
    html_parts = []

    if '<|think|>' in content and '<|end_think|>' in content:
        m = re.search(r'<\|think\|>(.*?)<\|end_think\|>', content, re.DOTALL)
        if m:
            think_text = _strip_special(m.group(1))
            if think_text:
                html_parts.append(format_thinking_html(
                    think_text.replace("\n", "<br>"), finished=True
                ))
            content = content[:m.start()] + content[m.end():]
    elif '<|think|>' in content and '<|end_think|>' not in content:
        m = re.search(r'<\|think\|>(.*)', content, re.DOTALL)
        if m:
            think_text = _strip_special(m.group(1))
            if think_text:
                html_parts.append(format_thinking_html(
                    think_text.replace("\n", "<br>"), finished=False
                ))
            content = content[:m.start()]

    tool_call_pattern = re.findall(r'ALSE(.*?)ALND', content, re.DOTALL)
    if tool_call_pattern:
        content = re.sub(r'ALSE.*?ALND', '', content, flags=re.DOTALL)
        for tc_raw in tool_call_pattern:
            tc_raw = tc_raw.strip()
            if not tc_raw:
                continue
            try:
                tc = json.loads(tc_raw)
                name = tc.get("name", "unknown")
                args_str = json.dumps(tc.get("arguments", {}), ensure_ascii=False)
                html_parts.append(format_tool_call_html(name, args_str))
            except json.JSONDecodeError:
                html_parts.append(format_tool_call_html("raw", tc_raw))

    cleaned = _strip_special(content)
    if cleaned:
        html_parts.append(cleaned)

    return "\n".join(html_parts) if html_parts else ""


def scan_weight_configs():
    configs = {}
    if not os.path.isdir(WEIGHT_DIR):
        return configs
    for fname in os.listdir(WEIGHT_DIR):
        if fname.endswith('.pth'):
            parts = fname.replace('.pth', '').split('_')
            if len(parts) >= 3:
                prefix = parts[0]
                try:
                    hidden = int(parts[-2])
                    layers = int(parts[-1])
                    label = f"{prefix} (H={hidden}, L={layers})"
                    path = os.path.join(WEIGHT_DIR, fname)
                    configs[label] = {
                        "path": path,
                        "prefix": prefix,
                        "hidden": hidden,
                        "layers": layers,
                    }
                except ValueError:
                    pass
    return configs


@st.cache_resource
def load_model_and_tokenizer(weight_path, hidden_size, num_layers, num_heads):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    voc = OpenASHVoc(agent_voc_path=agent_voc_path)
    voc_size = len(voc.token_to_id) + 1
    model = OpenASH(
        voc_size=voc_size, hidden_size=hidden_size,
        num_heads=num_heads, num_layers=num_layers,
        model_flag="infer"
    )
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True), strict=False)
    model = model.to(device).eval()
    return model, voc, device


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def render_user_bubble(text):
    st.markdown(
        f'<div style="display: flex; justify-content: flex-end;">'
        f'<div style="display: inline-block; margin: 10px 0; padding: 8px 12px; '
        f'background-color: #3d4450; border-radius: 22px; color: white;">{text}</div></div>',
        unsafe_allow_html=True
    )


def render_assistant_message(content):
    html = process_assistant_content(content)
    if html:
        st.markdown(html, unsafe_allow_html=True)


def render_history(messages):
    for msg in messages:
        if msg["role"] == "user":
            render_user_bubble(msg["content"])
        elif msg["role"] == "assistant":
            render_assistant_message(msg["content"])


def main():
    weight_configs = scan_weight_configs()

    if not weight_configs:
        default_label = f"full_sft (H={HIDDEN_SIZE}, L={NUM_LAYERS})"
        default_path = os.path.join(WEIGHT_DIR, f"full_sft_{HIDDEN_SIZE}_{NUM_LAYERS}.pth")
        weight_configs[default_label] = {
            "path": default_path,
            "prefix": "full_sft",
            "hidden": HIDDEN_SIZE,
            "layers": NUM_LAYERS,
        }

    selected_label = st.sidebar.selectbox('Model', list(weight_configs.keys()), index=0)
    cfg = weight_configs[selected_label]

    st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

    lang_options = {'中文': 'zh', 'English': 'en'}
    current_lang = st.session_state.get('lang', 'en')
    lang_index = 0 if current_lang == 'zh' else 1
    lang_label = st.sidebar.radio('Language / 语言', list(lang_options.keys()), index=lang_index, horizontal=True)
    if lang_options[lang_label] != current_lang:
        st.session_state.lang = lang_options[lang_label]
        st.rerun()

    st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

    st.session_state.history_chat_num = st.sidebar.slider(get_text('history_rounds'), 0, 8, 0, step=2)
    st.session_state.max_new_tokens = st.sidebar.slider(get_text('max_length'), 128, 40960, 1024, step=1)
    st.session_state.temperature = st.sidebar.slider(get_text('temperature'), 0.1, 1.5, 0.5, step=0.01)
    st.session_state.top_k = st.sidebar.slider(get_text('top_k'), 1, 100, 30, step=1)
    st.session_state.top_p = st.sidebar.slider(get_text('top_p'), 0.5, 1.0, 0.85, step=0.01)
    st.session_state.repetition_penalty = st.sidebar.slider(get_text('repetition_penalty'), 1.0, 2.0, 1.35, step=0.01)

    st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

    st.session_state.enable_thinking = st.sidebar.checkbox(
        get_text('thinking'), value=False, help=get_text('think_tip')
    )
    st.session_state.show_raw = st.sidebar.checkbox('Debug (Raw Output)', value=False)
    st.session_state.selected_tools = []
    with st.sidebar.expander(get_text('tools')):
        st.caption(get_text('tool_select'))
        selected_count = sum(
            1 for tool in TOOLS if st.session_state.get(f"tool_{tool['function']['name']}", False)
        )
        for tool in TOOLS:
            name = tool['function']['name']
            short_name = TOOL_SHORT_NAMES.get(name, name)
            checked = st.checkbox(
                short_name, key=f"tool_{name}",
                disabled=(selected_count >= 4 and not st.session_state.get(f"tool_{name}", False))
            )
            if checked and len(st.session_state.selected_tools) < 4:
                st.session_state.selected_tools.append(name)

    lang = st.session_state.get('lang', 'en')
    slogan = (
        f"我是 OpenASH，有什么可以帮你的？"
        if lang == 'zh'
        else "I am OpenASH, how can I help you?"
    )

    st.markdown(
        f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
        f'<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; '
        f'align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
        f'<span style="font-size: 26px; margin-left: 10px;">&#9883; {slogan}</span>'
        f'</div>'
        f'<span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">'
        f'{get_text("disclaimer")}</span></div>',
        unsafe_allow_html=True
    )

    model, tokenizer, device = load_model_and_tokenizer(
        cfg["path"], cfg["hidden"], cfg["layers"], NUM_HEADS
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    messages = st.session_state.messages

    render_history(messages)

    prompt = st.chat_input(key="input", placeholder=get_text('send'))

    if prompt:
        render_user_bubble(prompt)
        messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})
        st.session_state.chat_messages.append(
            {"role": "user", "content": prompt[-st.session_state.max_new_tokens:]}
        )

        placeholder = st.empty()

        setup_seed(random.randint(0, 2**32 - 1))

        tools_json = None
        selected_tool_names = st.session_state.get('selected_tools', [])
        if selected_tool_names:
            tools_json = json.dumps(
                [t for t in TOOLS if t['function']['name'] in selected_tool_names],
                ensure_ascii=False
            )

        sys_content = (
            "你是一个有用的AI助手。请用完整且友好的方式回答用户问题。"
            if lang == 'zh'
            else "You are a helpful AI assistant."
        )
        sys_msg = {"role": "system", "content": sys_content}
        if tools_json:
            sys_msg["tools"] = tools_json

        chat_history = (
            [sys_msg]
            + st.session_state.chat_messages[-(st.session_state.history_chat_num + 1):]
        )

        sp = _sp(tokenizer)
        prompt_ids = build_chat_prompt(tokenizer, chat_history, tools=tools_json)
        prompt_ids += [sp["im_start"], sp["agent"]]

        full_text = ""
        full_ids = []
        for text_chunk, ids_batch in generate_stream(
            model, tokenizer, prompt_ids,
            max_new_tokens=st.session_state.max_new_tokens,
            temperature=st.session_state.temperature,
            top_k=st.session_state.top_k,
            top_p=st.session_state.top_p,
            repetition_penalty=st.session_state.repetition_penalty,
        ):
            full_text += text_chunk
            full_ids = ids_batch
            if st.session_state.get('show_raw', False):
                placeholder.text(full_text)
            else:
                placeholder.markdown(
                    process_assistant_content(full_text, is_streaming=True),
                    unsafe_allow_html=True
                )

        result = format_response(tokenizer, full_ids)

        tool_calls_str = result.get("tool_calls", "")
        if tool_calls_str:
            try:
                tool_calls_parsed = json.loads(tool_calls_str)
                if isinstance(tool_calls_parsed, dict):
                    tool_calls_parsed = [tool_calls_parsed]
                for tc in tool_calls_parsed:
                    if "function" in tc and isinstance(tc["function"], dict):
                        tc_name = tc["function"].get("name", "")
                        tc_args_raw = tc["function"].get("arguments", {})
                        if isinstance(tc_args_raw, str):
                            try:
                                tc_args = json.loads(tc_args_raw)
                            except json.JSONDecodeError:
                                tc_args = {}
                        else:
                            tc_args = tc_args_raw
                    else:
                        tc_name = tc.get("name", "")
                        tc_args = tc.get("arguments", {})
                    tool_result = execute_tool(tc_name, tc_args)
                    result_str = json.dumps(tool_result, ensure_ascii=False)
                    full_text += "\n" + format_tool_result_html(tc_name, result_str)
                    placeholder.markdown(
                        process_assistant_content(full_text),
                        unsafe_allow_html=True
                    )
            except json.JSONDecodeError:
                pass

        processed = process_assistant_content(full_text)
        if not processed:
            raw_preview = _SPECIAL_TOKEN_RE.sub('', full_text).strip()
            if raw_preview:
                processed = raw_preview
            else:
                processed = '<div style="color: #999; font-style: italic;">（模型未生成有效回复）</div>'

        messages.append({"role": "assistant", "content": full_text})
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": result.get("content", ""),
            "reasoning_content": result.get("thinking", ""),
            "tool_calls": result.get("tool_calls", ""),
        })

        placeholder.markdown(processed, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
