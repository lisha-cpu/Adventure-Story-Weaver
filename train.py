import argparse, os, torch
from tokenizer import BPETokenizer
from model import MiniGPT

def main():
    p=argparse.ArgumentParser(description="Train Adventure Story Weaver")
    p.add_argument("--epochs",type=int,default=1500)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--block-size",type=int,default=128)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--merges",type=int,default=300)
    p.add_argument("--steps",type=int,default=1500)
    args=p.parse_args()
    torch.manual_seed(42)
    text=open("corpus.txt",encoding="utf-8").read()
    tokenizer=BPETokenizer().train(text,args.merges)
    tokenizer.save("tokenizer.json")
    data=torch.tensor(tokenizer.encode(text),dtype=torch.long)
    split=int(0.9*len(data)); train_data=data[:split]; val_data=data[split:]
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=MiniGPT(tokenizer.vocab_size,args.block_size).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    def batch(source):
        if len(source)<=args.block_size+1: source=train_data
        starts=torch.randint(0,len(source)-args.block_size-1,(args.batch_size,))
        x=torch.stack([source[i:i+args.block_size] for i in starts])
        y=torch.stack([source[i+1:i+args.block_size+1] for i in starts])
        return x.to(device),y.to(device)
    model.train()
    for step in range(args.steps):
        x,y=batch(train_data); _,loss=model(x,y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if step%100==0:
            model.eval()
            with torch.no_grad():
                vx,vy=batch(val_data if len(val_data)>args.block_size+1 else train_data)
                _,vl=model(vx,vy)
            print(f"step {step:4d} | train loss {loss.item():.3f} | validation {vl.item():.3f}")
            model.train()
    os.makedirs("artifacts",exist_ok=True)
    torch.save({"model":model.state_dict(),"vocab_size":tokenizer.vocab_size,
                "block_size":args.block_size,"n_embd":128,"n_head":4,"n_layer":3},"artifacts/model.pt")
    print("Training finished. Saved artifacts/model.pt and tokenizer.json")
if __name__=="__main__": main()
