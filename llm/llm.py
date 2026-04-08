import os
import tiktoken
import torch
import requests
import json
from copy import deepcopy
from transformers import pipeline
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
MAX_NEW_TOKENS = 16384

class LLMBase:
    def __init__(self, temp: float = 0., top_p: float = 1.):
        self.temperature = temp
        self.top_p = top_p
        self.prompt_chain = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset(self):
        self.prompt_chain = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def count_tokens(self, string: str):
        pass

    def init_prompt_chain(self, content: str, prompt: str):
        pass

    def update_prompt_chain(self, content: str, prompt: str):
        pass

    def update_prompt_chain_w_response(self, response: str, role: str = "assistant"):
        self.prompt_chain.append({"role": role, "content": response})

    def query(self, content: str, prompt: str):
        pass

    def query_msg_chain(self):
        pass

    @staticmethod
    def log(context: str, save_name: str):
        with open(save_name, "w") as f:
            f.write(context)

class Llama3(LLMBase):
    def __init__(self, model_name: str, temp: float = 0., top_p: float = 1.):
        super().__init__(temp, top_p)
        self.model_id = "meta-llama/Meta-{}".format(model_name)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.bfloat16
                },
                "low_cpu_mem_usage": True,
            },
            device_map="auto"
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    def count_tokens(self, string: str):
        tokens = self.pipeline.tokenizer.tokenize(string)
        return len(tokens)

    def init_prompt_chain(self, content: str, prompt: str):
        assert len(self.prompt_chain) == 0, "Prompt chain is not empty!"
        self.prompt_chain.extend([{"role": "system", "content": content},
                                  {"role": "user", "content": prompt}])

    def update_prompt_chain(self, content: str, prompt: str):
        if len(self.prompt_chain) > 0:
            self.prompt_chain[0]["content"] = content
        self.prompt_chain.append({"role": "user", "content": prompt})

    def query(self, content: str, prompt: str):
        messages = [
            {"role": "system", "content": content},
            {"role": "user", "content": prompt}
        ]
        response = self.pipeline(
            messages,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=self.terminators,
            do_sample=True,
        )
        output = response[0]["generated_text"][-1]['content']
        torch.cuda.empty_cache()
        return output

    def query_msg_chain(self):
        response = self.pipeline(
            self.prompt_chain,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=self.terminators,
            do_sample=True,
        )
        output = response[0]["generated_text"][-1]['content']
        torch.cuda.empty_cache()
        return output

class GPT(LLMBase):
    def __init__(self, model_name: str, temp: float = 0., top_p: float = 1.):
        super().__init__(temp, top_p)
        self.model_id = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def count_tokens(self, string: str):
        encoding_name = deepcopy(self.model_id)
        if "gpt-35" in encoding_name:
            encoding_name = encoding_name.replace("gpt-35", "gpt-3.5")
        try:
            encoding = tiktoken.encoding_for_model(encoding_name)
        except:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(string))

    def init_prompt_chain(self, content: str, prompt: str):
        assert len(self.prompt_chain) == 0, "Prompt chain is not empty!"
        self.prompt_chain.extend([{"role": "system", "content": content},
                                  {"role": "user", "content": prompt}])

    def update_prompt_chain(self, content: str, prompt: str):
        if len(self.prompt_chain) > 0:
            self.prompt_chain[0]["content"] = content
        self.prompt_chain.append({"role": "user", "content": prompt})

    def query(self, content: str, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model_id,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": content},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def query_msg_chain(self):
        response = self.client.chat.completions.create(
            model=self.model_id,
            temperature=self.temperature,
            messages=self.prompt_chain
        )
        return response.choices[0].message.content

class GeminiModel(LLMBase):
    def __init__(self, model_name: str, temp: float = 0., top_p: float = 1.0):
        super().__init__(temp, top_p)
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")
        
    def _get_valid_model(self):
        """APIに直接『使えるモデル一覧』を問い合わせて、確実に存在する名前を取得する"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                models = response.json().get('models', [])
                
                # generateContentをサポートし、名前にgeminiを含むものを探す
                available = [
                    m['name'] for m in models 
                    if 'generateContent' in m.get('supportedGenerationMethods', []) 
                    and 'gemini' in m['name'].lower()
                ]
                
                if available:
                    # 'flash'が含まれるものを優先的に選択
                    for m in available:
                        if "flash" in m:
                            return m
                    return available[0]
            else:
                print(f"[API ERROR] ListModels failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"[API ERROR] Failed to fetch model list: {e}")
            
        # 取得に失敗した場合は推測でフォールバック
        return "models/gemini-1.5-flash"

    def init_prompt_chain(self, content: str, prompt: str):
        self.prompt_chain = [
            {"role": "user", "content": f"System Instruction:\n{content}\n\nUser Request:\n{prompt}"}
        ]

    def update_prompt_chain(self, content: str, prompt: str):
        if len(self.prompt_chain) > 0:
            self.prompt_chain[0]["content"] = f"System Instruction:\n{content}\n\nUser Request:\n{prompt}"
        else:
            self.init_prompt_chain(content, prompt)

    def query_msg_chain(self):
        # 探索通信を完全に削除し、判明したモデル名を直打ちする
        actual_model_name = "models/gemini-2.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/{actual_model_name}:generateContent?key={self.api_key}"

        contents = []
        for msg in self.prompt_chain:
            role = "user" if msg["role"] in ["user", "system"] else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "topP": self.top_p,
                "maxOutputTokens": 4096
            }
        }
        headers = {'Content-Type': 'application/json'}

        # 推論APIのみをダイレクトに叩く
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            res_json = response.json()
            output_text = res_json['candidates'][0]['content']['parts'][0]['text']
            self.prompt_chain.append({"role": "model", "content": output_text})
            return output_text
        else:
            raise Exception(f"Gemini API Error: {response.status_code} - {response.text}")
            
def load_llm(model_name: str, temp: float = 0., top_p: float = 1.):
    m_lower = model_name.lower()
    if "llama" in m_lower:
        return Llama3(model_name, temp, top_p)
    elif "gpt" in m_lower:
        return GPT(model_name, temp, top_p)
    elif "gemini" in m_lower:
        return GeminiModel(model_name, temp, top_p)
    else:
        raise Exception(f"Invalid model name: {model_name}")