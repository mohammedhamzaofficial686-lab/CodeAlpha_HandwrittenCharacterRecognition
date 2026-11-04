"""Skribl local web server. Starts one process with a background feedback worker."""
import json
import logging
import os
import secrets
import threading
import uuid
import warnings
from flask import Flask, jsonify, render_template, request, session
from PIL import Image
import torch
from model import ROOT, MODELS, LABELS, load_model
from preprocessing import segment, tensor_images
from feedback import INSTANCE, remember, confirm, start_worker

warnings.simplefilter("error",Image.DecompressionBombWarning)
torch.set_num_threads(2)
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
app=Flask(__name__, template_folder="frontend", static_folder="frontend", static_url_path="/static")
secret_file=INSTANCE/"session.key"
if not secret_file.exists(): secret_file.write_text(secrets.token_hex(32),encoding="ascii")
app.config.update(SECRET_KEY=os.environ.get("SKRIBL_SECRET_KEY") or secret_file.read_text().strip(),MAX_CONTENT_LENGTH=9*1024*1024,SESSION_COOKIE_HTTPONLY=True,SESSION_COOKIE_SAMESITE="Lax")
cache={};model_lock=threading.Lock()

def active_model(mode):
    file=MODELS/f"{mode}.pt"
    if not file.exists(): raise FileNotFoundError(f"The {mode} model is not trained yet. Run train.py first.")
    stamp=file.stat().st_mtime_ns
    with model_lock:
        if mode not in cache or cache[mode][0]!=stamp:
            model,checkpoint=load_model(mode);cache[mode]=(stamp,model,checkpoint)
        return cache[mode][1:]

@app.before_request
def protect_writes():
    if request.method=="POST":
        supplied=request.headers.get("X-CSRF-Token","")
        if not supplied or not secrets.compare_digest(supplied,session.get("csrf","")):
            return jsonify(error="Session expired. Refresh the page and try again."),403

@app.after_request
def security_headers(response):
    response.headers["X-Content-Type-Options"]="nosniff"
    response.headers["X-Frame-Options"]="DENY"
    response.headers["Referrer-Policy"]="same-origin"
    return response

@app.get("/")
@app.get("/digits")
@app.get("/letters")
def index():
    session.setdefault("csrf",secrets.token_hex(24))
    return render_template("index.html",mode=request.path.strip("/"),csrf=session["csrf"])

@app.get("/api/status")
def status():
    return jsonify(models={m:(MODELS/f"{m}.pt").exists() for m in LABELS})

@app.post("/api/predict")
def predict():
    try:
        mode=request.form.get("mode","")
        if mode not in LABELS: raise ValueError("Choose Digits or Letters.")
        upload=request.files.get("image")
        if not upload: raise ValueError("Draw or upload an image first.")
        expected_text=request.form.get("expected","")
        expected=int(expected_text) if expected_text else None
        if expected is not None and not 1<=expected<=20: raise ValueError("Expected count must be 1 to 20.")
        model,checkpoint=active_model(mode)
        glyphs,boxes=segment(upload.read(),expected)
        with torch.inference_mode():
            probabilities=model(tensor_images(glyphs)).softmax(1)
        results=[]
        for probability,box in zip(probabilities,boxes):
            values,indices=probability.topk(3)
            guesses=[{"character":LABELS[mode][int(i)],"confidence":round(float(v)*100,2)} for v,i in zip(values,indices)]
            results.append({"answer":guesses[0]["character"],"guesses":guesses,"box":box})
        identifier=uuid.uuid4().hex
        remember(identifier,mode,glyphs)
        owned=session.get("predictions",[]);session["predictions"]=(owned+[identifier])[-10:]
        return jsonify(id=identifier,answer="".join(r["answer"] for r in results),characters=results,count=len(results),model_version=checkpoint["version"])
    except FileNotFoundError as exc: return jsonify(error=str(exc)),409
    except (ValueError,Image.DecompressionBombWarning,Image.DecompressionBombError) as exc: return jsonify(error=str(exc)),400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction failed. Check the server terminal for details."),500

@app.post("/api/feedback")
def feedback():
    body=request.get_json(silent=True) or {}
    identifier=body.get("id","")
    if identifier not in session.get("predictions",[]): return jsonify(error="Recognize this image again before confirming."),403
    if body.get("confirmed") is not True: return jsonify(error="Confirm the answer before saving it."),400
    try:
        count=confirm(identifier,body.get("answer",""))
        return jsonify(message="Thanks! Your confirmed example was saved." if count else "Thanks! This example is already in the learning collection.")
    except ValueError as exc: return jsonify(error=str(exc)),400

@app.get("/api/metrics")
def metrics():
    reports={}
    for mode in LABELS:
        file=ROOT/"reports"/f"{mode}_metrics.json"
        if file.exists():
            data=json.loads(file.read_text(encoding="utf-8"))
            reports[mode]={k:data[k] for k in ("test_accuracy","test_count","created_at","test_sampling")}
    return jsonify(reports)

@app.errorhandler(413)
def too_large(error): return jsonify(error="Image is too large. Use a file under 8 MB."),413

if __name__=="__main__":
    from waitress import serve
    start_worker()
    print("Skribl is ready at http://127.0.0.1:5000",flush=True)
    serve(app,host=os.environ.get("SKRIBL_HOST","127.0.0.1"),port=int(os.environ.get("PORT","5000")),threads=4)
