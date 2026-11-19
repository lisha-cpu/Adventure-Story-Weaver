import argparse, torch
from tokenizer import BPETokenizer
from model import MiniGPT

def main():
    p=argparse.ArgumentParser(description="Generate adventure stories")
    p.add_argument("--prompt",default=None)
    p.add_argument("--tokens",type=int,default=300)
    p.add_argument("--temperature",type=float,default=0.9)
    p.add_argument("--top-k",type=int,default=40)
    args=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    tokenizer=BPETokenizer.load("tokenizer.json")
    ckpt=torch.load("artifacts/model.pt",map_location=device)
    model=MiniGPT(ckpt["vocab_size"],ckpt["block_size"],ckpt["n_embd"],ckpt["n_head"],ckpt["n_layer"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    def generate(prompt):
        ids=tokenizer.encode(prompt)
        if not ids: ids=[ord(" ")]
        x=torch.tensor([ids],dtype=torch.long,device=device)
        with torch.no_grad():
            out=model.generate(x,args.tokens,args.temperature,args.top_k)
        print("\n"+tokenizer.decode(out[0].tolist())+"\n")
    if args.prompt:
        generate(args.prompt); return
    print("Adventure Story Weaver — enter a prompt, or type quit.")
    while True:
        prompt=input("\nYour story prompt> ").strip()
        if prompt.lower() in {"quit","exit"}: break
        if not prompt: continue
        generate(prompt)
if __name__=="__main__": main()
