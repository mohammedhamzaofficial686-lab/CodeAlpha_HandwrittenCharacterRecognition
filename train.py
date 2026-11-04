"""Train and evaluate Skribl. Run: python train.py --mode both --epochs 5."""
from filelock import FileLock
import argparse
import csv
import json
import time
from datetime import datetime, timezone
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST, EMNIST
from model import ROOT, MODELS, LABELS, CharacterCNN, atomic_save
from preprocessing import normalize_glyph

SEED = 42

def prepare(mode, train, per_class):
    source = (MNIST(ROOT / "data", train=train, download=True) if mode=="digits" else
              EMNIST(ROOT / "data", split="byclass", train=train, download=True))
    labels = source.targets.numpy().copy()
    if mode=="letters":
        labels -= 10
    rng = np.random.default_rng(SEED if train else SEED+1)
    selected=[]
    for c in range(len(LABELS[mode])):
        rows=np.flatnonzero(labels==c); rng.shuffle(rows)
        selected.extend(rows[:per_class] if per_class else rows)
    rng.shuffle(selected)
    images=[]; kept_labels=[]
    print(f"Normalizing {len(selected):,} {mode} images ({'train' if train else 'test'})...", flush=True)
    for i in selected:
        image=source.data[int(i)].numpy()
        if mode=="letters": image=image.T
        try:
            images.append(normalize_glyph(image));kept_labels.append(int(labels[i]))
        except ValueError:
            continue
    x=torch.from_numpy(np.stack(images)[:,None].astype(np.float32)/255)
    y=torch.tensor(kept_labels, dtype=torch.long)
    return x,y

def evaluate(model,x,y,batch=256):
    model.eval(); predicted=[]; loss=0.0
    with torch.inference_mode():
        for xb,yb in DataLoader(TensorDataset(x,y),batch_size=batch):
            output=model(xb); loss+=nn.functional.cross_entropy(output,yb,reduction="sum").item()
            predicted.append(output.argmax(1))
    pred=torch.cat(predicted)
    return float((pred==y).float().mean()), loss/len(y), pred

def train_mode(mode,args):
    torch.manual_seed(SEED)
    x,y=prepare(mode,True,args.per_class)
    # Stratified validation comes only from training data; official test is untouched.
    generator=torch.Generator().manual_seed(SEED)
    tr,va=[],[]
    for c in range(len(LABELS[mode])):
        indices=torch.where(y==c)[0]; indices=indices[torch.randperm(len(indices),generator=generator)]
        n=max(1,len(indices)//10); va.extend(indices[:n].tolist());tr.extend(indices[n:].tolist())
    tx,ty,vx,vy=x[tr],y[tr],x[va],y[va]
    del x,y
    model=CharacterCNN(len(LABELS[mode]))
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0001)
    loader=DataLoader(TensorDataset(tx,ty),batch_size=args.batch_size,shuffle=True,generator=generator)
    history=[]; best=-1; best_state=None; start=time.monotonic()
    for epoch in range(1,args.epochs+1):
        model.train(); total=correct=count=0
        for xb,yb in loader:
            optimizer.zero_grad(set_to_none=True); output=model(xb)
            loss=nn.functional.cross_entropy(output,yb);loss.backward();optimizer.step()
            total+=loss.item()*len(yb);correct+=(output.argmax(1)==yb).sum().item();count+=len(yb)
        accuracy,val_loss,_=evaluate(model,vx,vy)
        history.append(dict(epoch=epoch,train_loss=total/count,train_accuracy=correct/count,val_loss=val_loss,val_accuracy=accuracy))
        print(f"{mode} | epoch {epoch}/{args.epochs} | train {correct/count:.2%} | validation {accuracy:.2%}",flush=True)
        if accuracy>best:
            best=accuracy;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    ex,ey=prepare(mode,False,args.test_per_class)
    test_accuracy,test_loss,pred=evaluate(model,ex,ey)
    classes=len(LABELS[mode]); confusion=torch.bincount(ey*classes+pred,minlength=classes*classes).reshape(classes,classes)
    stamp=datetime.now(timezone.utc).isoformat()
    checkpoint={"state_dict":best_state,"labels":LABELS[mode],"version":stamp,"source":"initial training","validation_accuracy":best,"base_test_accuracy":test_accuracy}
    atomic_save(checkpoint,MODELS/f"{mode}.pt")
    # Replay and validation tensors are disjoint; only replay enters optimizer steps.
    rt=torch.randperm(len(ty),generator=generator)[:3000]
    rv=torch.randperm(len(vy),generator=generator)[:3000]
    atomic_save({"train_x":tx[rt],"train_y":ty[rt],"val_x":vx[rv],"val_y":vy[rv]},MODELS/f"{mode}_replay.pt")
    results=ROOT/"reports";results.mkdir(exist_ok=True)
    report={"mode":mode,"created_at":stamp,"seed":SEED,"train_count":len(ty),"validation_count":len(vy),"test_count":len(ey),"test_sampling":"all" if args.test_per_class==0 else f"up to {args.test_per_class} per class", "test_accuracy":test_accuracy,"test_loss":test_loss,"training_seconds":round(time.monotonic()-start,1),"history":history,"labels":LABELS[mode],"confusion_matrix":confusion.tolist()}
    (results/f"{mode}_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (results/f"{mode}_history.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot([h["train_accuracy"] for h in history],label="Train");axes[0].plot([h["val_accuracy"] for h in history],label="Validation");axes[0].set_title("Accuracy by epoch");axes[0].legend()
    axes[1].plot([h["train_loss"] for h in history],label="Train");axes[1].plot([h["val_loss"] for h in history],label="Validation");axes[1].set_title("Loss by epoch");axes[1].legend()
    fig.tight_layout();fig.savefig(results/f"{mode}_training.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,10));im=ax.imshow(confusion.numpy(),cmap="Greens")
    ax.set(xticks=range(classes),yticks=range(classes),xticklabels=LABELS[mode],yticklabels=LABELS[mode],xlabel="Predicted",ylabel="True",title=f"{mode.title()} test confusion matrix")
    ax.tick_params(labelsize=7);fig.colorbar(im);fig.tight_layout();fig.savefig(results/f"{mode}_confusion.png",dpi=150);plt.close(fig)
    print(f"Saved {mode} model. Test accuracy: {test_accuracy:.2%} on {len(ey):,} held-out examples.",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["digits","letters","both"],default="both")
    parser.add_argument("--epochs",type=int,default=5)
    parser.add_argument("--per-class",type=int,default=2000,help="Maximum training examples/class; 0 uses all")
    parser.add_argument("--test-per-class",type=int,default=200,help="Maximum test examples/class; 0 uses all")
    parser.add_argument("--batch-size",type=int,default=128)
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.threads<1 or args.per_class<0 or args.test_per_class<0 or 0<args.per_class<20:
        parser.error("Use positive epochs/batch/threads and per-class 0 or >=20.")
    torch.set_num_threads(args.threads)
    with FileLock(str(MODELS/"training.lock"),timeout=0):
        for mode in (["digits","letters"] if args.mode=="both" else [args.mode]): train_mode(mode,args)
