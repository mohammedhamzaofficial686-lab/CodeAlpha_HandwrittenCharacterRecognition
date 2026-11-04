"""Local feedback persistence and conservative background fine-tuning."""
from contextlib import contextmanager
from filelock import FileLock, Timeout
import hashlib
import io
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import ROOT, MODELS, LABELS, load_model, atomic_save
from preprocessing import tensor_images

log=logging.getLogger(__name__)
INSTANCE=ROOT/"instance"; INSTANCE.mkdir(exist_ok=True)
DB=INSTANCE/"feedback.sqlite3"
worker_lock=threading.Lock()
THRESHOLD=max(10,int(os.environ.get("SKRIBL_RETRAIN_EVERY","25")))

@contextmanager
def connect():
    db=sqlite3.connect(DB,timeout=20)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS examples (digest TEXT PRIMARY KEY, mode TEXT NOT NULL, image BLOB NOT NULL, label INTEGER NOT NULL, holdout INTEGER NOT NULL, processed INTEGER NOT NULL DEFAULT 0)")
    db.execute("CREATE TABLE IF NOT EXISTS predictions (id TEXT PRIMARY KEY, mode TEXT NOT NULL, images BLOB NOT NULL, created REAL NOT NULL, used INTEGER NOT NULL DEFAULT 0)")
    db.execute("CREATE TABLE IF NOT EXISTS runs (created TEXT, mode TEXT, accepted INTEGER, details TEXT)")
    try:
        with db:
            yield db
    finally:
        db.close()

def remember(identifier,mode,images):
    stream=io.BytesIO(); np.save(stream,np.stack(images),allow_pickle=False)
    with connect() as db:
        db.execute("DELETE FROM predictions WHERE created < ?",(time.time()-3600,))
        db.execute("INSERT INTO predictions VALUES (?,?,?,?,0)",(identifier,mode,stream.getvalue(),time.time()))

def confirm(identifier,text):
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row=db.execute("SELECT mode,images,used,created FROM predictions WHERE id=?",(identifier,)).fetchone()
        if not row or row[3]<time.time()-3600: raise ValueError("This result expired. Recognize the image again.")
        if row[2]: raise ValueError("This example has already been confirmed.")
        mode,raw,_,_=row
        images=np.load(io.BytesIO(raw),allow_pickle=False)
        if not isinstance(text,str) or len(text)!=len(images) or any(c not in LABELS[mode] for c in text):
            raise ValueError(f"Enter exactly {len(images)} {'digits' if mode=='digits' else 'English letters'}, preserving uppercase/lowercase. If the count is wrong, redraw with clearer gaps.")
        added=0
        for image,c in zip(images,text):
            digest=hashlib.sha256(mode.encode()+image.tobytes()).hexdigest()
            holdout=int(int(digest[:8],16)%5==0)
            cursor=db.execute("INSERT OR IGNORE INTO examples(digest,mode,image,label,holdout) VALUES (?,?,?,?,?)",(digest,mode,image.tobytes(),LABELS[mode].index(c),holdout))
            added+=cursor.rowcount
        db.execute("UPDATE predictions SET used=1 WHERE id=?",(identifier,))
    return added

def scores(model,x,y):
    correct=total=0
    model.eval()
    with torch.inference_mode():
        for xb,yb in DataLoader(TensorDataset(x,y),batch_size=128):
            correct+=(model(xb).argmax(1)==yb).sum().item(); total+=len(yb)
    return correct/total

def retrain(mode):
    if not (MODELS/f"{mode}_replay.pt").exists(): return
    with connect() as db:
        pending=db.execute("SELECT COUNT(*) FROM examples WHERE mode=? AND processed=0",(mode,)).fetchone()[0]
        if pending<THRESHOLD: return
        rows=db.execute("SELECT digest,image,label,holdout FROM examples WHERE mode=? ORDER BY rowid DESC LIMIT 4000",(mode,)).fetchall()
    training=[r for r in rows if not r[3]]; validation=[r for r in rows if r[3]]
    if len(training)<5 or len(validation)<5: return
    x=tensor_images([np.frombuffer(r[1],dtype=np.uint8).reshape(28,28) for r in training])
    y=torch.tensor([r[2] for r in training])
    vx=tensor_images([np.frombuffer(r[1],dtype=np.uint8).reshape(28,28) for r in validation])
    vy=torch.tensor([r[2] for r in validation])
    model,checkpoint=load_model(mode)
    replay=torch.load(MODELS/f"{mode}_replay.pt",map_location="cpu",weights_only=True)
    before_base=scores(model,replay["val_x"],replay["val_y"])
    before_feedback=scores(model,vx,vy)
    rx,ry=replay["train_x"],replay["train_y"]
    combined=TensorDataset(torch.cat((rx,x.repeat(3,1,1,1))),torch.cat((ry,y.repeat(3))))
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.00005)
    model.train()
    for xb,yb in DataLoader(combined,batch_size=64,shuffle=True):
        optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model(xb),yb);loss.backward();optimizer.step()
    after_base=scores(model,replay["val_x"],replay["val_y"])
    after_feedback=scores(model,vx,vy)
    accepted=after_base>=before_base and after_feedback>=before_feedback and (after_base>before_base or after_feedback>before_feedback)
    stamp=datetime.now(timezone.utc).isoformat()
    details={"base_before":before_base,"base_after":after_base,"feedback_before":before_feedback,"feedback_after":after_feedback,"feedback_train":len(training),"feedback_validation":len(validation)}
    if accepted:
        # Preserve the previous active checkpoint for rollback.
        atomic_save(checkpoint,MODELS/f"{mode}_previous.pt")
        checkpoint.update(state_dict=model.state_dict(),version=stamp,source="confirmed feedback",feedback_validation=details)
        atomic_save(checkpoint,MODELS/f"{mode}.pt")
    with connect() as db:
        db.executemany("UPDATE examples SET processed=1 WHERE digest=?",[(r[0],) for r in rows])
        db.execute("INSERT INTO runs VALUES (?,?,?,?)",(stamp,mode,int(accepted),json.dumps(details)))
    log.info("Feedback candidate %s: %s",mode,"accepted" if accepted else "kept existing model")

def work():
    if not worker_lock.acquire(blocking=False): return
    try:
        with connect() as db:
            db.execute("DELETE FROM predictions WHERE created < ?",(time.time()-3600,))
        try:
            with FileLock(str(MODELS/"training.lock"),timeout=0):
                for mode in LABELS:
                    try: retrain(mode)
                    except Exception: log.exception("Feedback training failed for %s; existing model retained",mode)
        except Timeout:
            pass
    finally: worker_lock.release()

def start_worker():
    if os.environ.get("SKRIBL_AUTO_RETRAIN","1")!="1": return
    def loop():
        while True:
            work();time.sleep(60)
    threading.Thread(target=loop,daemon=True,name="skribl-feedback").start()
