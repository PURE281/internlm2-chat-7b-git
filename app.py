import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
base_path = './my_dream'
os.system('apt install git')
os.system('apt install git-lfs')
os.system(f'git clone https://code.openxlab.org.cn/pure81/my_dream.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')
tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()
