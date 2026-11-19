import create_sequential_embeddings as c, torch, math
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained('openai-community/gpt2')
m = AutoModelForCausalLM.from_pretrained('openai-community/gpt2').to(c.device).eval()
p = 'def add(a, b):\n    return a + b'
_, ppl = c.generate_embeddings(p, m, tok, compute_ppl=True)
ids = tok(p, return_tensors='pt').input_ids.to(c.device)
with torch.no_grad(): ref = math.exp(m(ids, labels=ids).loss.item())
print(ppl, ref)   # should match: GPT-2 adds no special tokens here
print(c.generate_embeddings('x', m, tok, compute_ppl=True)[1])  # one-token prompt, should be nan