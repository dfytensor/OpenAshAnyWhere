import json
import jieba
import unicodedata
from tqdm import tqdm


class OpenASHVoc:

    def __init__(self, agent_voc_path="open_ash_voc_agent.json",
                 voc_path=r'vocabulary_nnn.json', voc_size=6400, two=100,
                 flag=False):
        if flag:
            self.gen_agent_voc_from_voc(voc_path=voc_path, voc_size=voc_size, two=two)
        with open(agent_voc_path, "r", encoding="utf-8") as config_file:
            voc = json.load(config_file)
        self.token_to_id = voc["token_to_id"]
        self.agent_token_to_id = voc["agent_token_to_token_id"]
        self.id_to_token = voc["id_to_token"]
        print("词表大小", len(self.token_to_id))
        self.voc_size = len(self.token_to_id)
        self.agent_token_id_to_token = voc["agent_token_id_to_token"]
        text = """测试代理词表"""
        if self.token_to_id.get("ts0") + 1 == self.token_to_id.get("ts1") and self.token_to_id.get(
                "ts1") + 1 == self.token_to_id.get(
            "ts2"):
            self.ts0 = self.token_to_id.get("ts0")
        else:
            return
        print("代理词表大小", len(self.token_to_id) - self.ts0)
        self._jieba = None
        self._jieba_warmup_text = text
        token_ids = self.encode(text)
        self.decode(token_ids)
        self.new_voc=[]

    @property
    def jieba(self):
        if self._jieba is None:
            self._jieba = jieba.Tokenizer()
            self._jieba.lcut(self._jieba_warmup_text)
        return self._jieba

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_jieba'] = None
        return state

    def is_meaningful(self, char):
        """严格定义：已分配 + 非控制字符"""
        try:

            cat = unicodedata.category(char)
            return not (cat.startswith('C') and cat not in ['Co', 'Cn'])
        except:
            return False

    def get_meaningful_chars(self):
        """获取有意义字符列表"""
        meaningful_chars = []
        for code in range(0x10000):  # 基本平面
            char = chr(code)
            if self.is_meaningful(char):
                meaningful_chars.append(char)
        return meaningful_chars[:-1]  # 移除最后一

    def gen_agent_voc_from_voc(self, voc_path='open_ash_voc.json', voc_size=6400, two=100):
        with open(voc_path, "r", encoding="utf-8") as f:
            vocabulary = json.load(f)
        chars=set(self.get_meaningful_chars())-(set(vocabulary["voc"]) &
         set(self.get_meaningful_chars()))
        vocabulary["voc"]+=chars
        total_voc_size = len(vocabulary["voc"])

        thr = int(total_voc_size ** 0.33)

        while total_voc_size - thr ** 3 < two ** 2 + voc_size:
            thr -= 1
        two = int((total_voc_size - thr ** 3) ** 0.5)
        while total_voc_size - thr ** 3 - two ** 2 < voc_size:
            two -= 1
        print(two * 2 + thr * 3 + total_voc_size - thr ** 3 - two ** 2)
        special_tokens = [
                             "<|pad|>", "<|im_start|>", "<|im_end|>", "<|think|>",
                             "<|end_think|>", "<|user|>", "<|agent|>", "<|system|>",
                             "<|func|>", "<|args|>", "<|unk|>",
                         ] + [
                             "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>",
                             "<|quad_start|>", "<|quad_end|>",
                             "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
                             "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<tts_pad>", "<tts_text_bos>",
                             "<tts_text_eod>", "<tts_text_bos_single>","<|tools|>","<|end_tools|>",
                             "<tool_call>", "</tool_call>",
                             "<tool_response>", "</tool_response>",
                         ]
        voc = vocabulary["voc"][:total_voc_size - thr ** 3 - two ** 2]
        voc += ["ts{}".format(i) for i in range(two)]
        voc += ["te{}".format(i) for i in range(two)]
        voc += ["rs{}".format(i) for i in range(thr)]
        voc += ["rc{}".format(i) for i in range(thr)]
        voc += ["re{}".format(i) for i in range(thr)]
        token_list = special_tokens + voc

        token_to_id = {token: i for i, token in enumerate(token_list)}
        id_to_token = {i: token for i, token in enumerate(token_list)}

        agent_token_id_to_token = dict()
        agent_token_to_token_id = dict()
        agent_token_id = []
        for i in tqdm(range(two)):
            for j in range(two):
                agent_token_id.append("{}t{}".format(i, j))

        for i in tqdm(range(thr)):
            for j in range(thr):
                for z in range(thr):
                    agent_token_id.append("{}t{}r{}".format(i, j, z))
        for token, token_id in zip(vocabulary["voc"][total_voc_size - thr ** 3 - two ** 2:], agent_token_id):
            agent_token_to_token_id[token] = token_id
            agent_token_id_to_token[token_id] = token
        with open('open_ash_voc_agent.json', "w", encoding="utf-8") as f:
            json.dump(
                {"agent_token_to_token_id": agent_token_to_token_id, "agent_token_id_to_token": agent_token_id_to_token,
                 "token_to_id": token_to_id, "id_to_token": id_to_token}, f, ensure_ascii=False, indent=4)

    def agent_encode(self, token):
        token_id = []
        agent_token_id = self.agent_token_to_id.get(token)
        if "r" in agent_token_id:
            rs, rc, re = agent_token_id.replace("t", "_").replace("r", "_").split("_")
            token_id.append(self.token_to_id.get("rs" + rs))
            token_id.append(self.token_to_id.get("rc" + rc))
            token_id.append(self.token_to_id.get("re" + re))

        else:
            ts, te = agent_token_id.split("t")
            token_id.append(self.token_to_id.get("ts" + ts))
            token_id.append(self.token_to_id.get("te" + te))
        return token_id

    def encode(self, text):
        text = self.jieba.lcut(text)
        token_id = []
        for token in text:
            if token in self.token_to_id:
                token_id.append(self.token_to_id.get(token))
            elif token in self.agent_token_to_id:
                token_id += self.agent_encode(token)
            else:
                stoken_id = []
                for stoken in list(token):
                    if stoken in self.token_to_id:
                        token_id.append(self.token_to_id.get(stoken))

                    elif stoken in self.agent_token_to_id:
                        stoken_id += self.agent_encode(stoken)
                    else:
                        self.new_voc.append(stoken)
                        stoken_id.append(10)

                token_id += stoken_id

        return token_id

    def decode(self, token_ids):
        text = ""
        agent_token_id = ""

        for token_id in token_ids:

            if token_id < int(self.ts0):
                token = self.id_to_token.get(str(token_id))
                if token:
                    text += token
            else:
                token = self.id_to_token.get(str(token_id))
                if "t" in token:
                    if "ts" in token:
                        agent_token_id = ""
                        agent_token_id += token.replace("ts", "")

                    elif "te" in token:
                        agent_token_id += "t" + token.replace("te", "")
                        agent_token = self.agent_token_id_to_token.get(agent_token_id)
                        if agent_token:
                            text += agent_token
                else:
                    if "rs" in token:
                        agent_token_id = ""
                        agent_token_id += token.replace("rs", "")
                    elif "rc" in token:
                        agent_token_id += "t" + token.replace("rc", "") + "r"
                    elif "re" in token:
                        agent_token_id += token.replace("re", "")
                        agent_token = self.agent_token_id_to_token.get(agent_token_id)
                        if agent_token:
                            text += agent_token
        return text


if __name__ == '__main__':
    # voc = OpenASHVoc(flag=True, voc_size=20000, two=200)
    voc = OpenASHVoc(flag=False, voc_size=20000, two=200)
    text="""写一首关于自然美景的诗歌。在绿野上，羚羊奔跑
鸟语花香在心头荡漾
涓涓小溪，蜿蜒、潺潺
绿树成荫，凉雨淅淅沥沥
自然的美景，如此神奇
让我们沉迷，无法自拔
在这美景之中，心灵得以宁静
如此小小的悦动，细腻而清新给我推荐一些旅游景点。如果您喜欢海滨风光，我推荐您去三亚、厦门或青岛；如果您喜欢历史文化，那么可以去西安、北京或者南京；如果您想度假放松，可以考虑去丽江、大理或者海南岛。当然，不同的景点有不同的特色和魅力，您还可以根据自己的兴趣和需要做进一步的选择。帮我想一些关于狗的名字。关于狗的名字，不同的主人有不同的喜好。如果您喜欢可爱、懒洋洋的小狗，可以考虑给您的宠物取小名，比如“豆豆”、“宝宝”等；如果您喜欢体型强壮而有力的狗狗，可以考虑一些响亮的名字，比如“大力”、“霸气”等；如果您对狗的品种有偏好，也可以考虑为它取一些相关的名字，比如“拉布拉多”、“雪橇犬”等。请给我介绍一下人工智能的技术和发展。人工智能的技术发展已经越来越快，包括深度学习、机器学习、自然语言处理、计算机视觉等。随着算力、数据量和算法的不断提高，人工智能也在不断地进步和演变。未来，人工智能将会被广泛运用于各个领域，包括医疗、金融、智能制造等。与此同时，人工智能也带来了伦理、隐私等方面的问题，需要我们权衡好利弊，做出正确的决策。
"""*2000
    print(len(text))
    t = voc.encode(text)
    print(len(t))
    # print(voc.decode(t))
