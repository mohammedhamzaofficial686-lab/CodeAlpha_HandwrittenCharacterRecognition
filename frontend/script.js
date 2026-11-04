'use strict';
const $ = id => document.getElementById(id);
const mode = document.body.dataset.mode;
const csrf = document.querySelector('meta[name="csrf-token"]').content;
$('aboutButton').addEventListener('click', async () => {
  $('projectSection').hidden = false;
  $('projectSection').scrollIntoView({behavior:'smooth'});
  try {
    const data = await (await fetch('/api/metrics')).json();
    $('metrics').replaceChildren();
    if (!Object.keys(data).length) $('metrics').textContent = 'Benchmark results appear after initial training. No accuracy is claimed before evaluation.';
    for (const [key,value] of Object.entries(data)) {
      const card=document.createElement('div');card.className='metric';
      const title=document.createElement('div');title.textContent=key==='digits'?'Digits · original benchmark':'Letters · original benchmark';
      const score=document.createElement('strong');score.textContent=(value.test_accuracy*100).toFixed(2)+'%';
      const detail=document.createElement('div');detail.textContent=value.test_count.toLocaleString()+' test images · '+value.test_sampling;
      card.append(title,score,detail);$('metrics').append(card);
    }
  } catch { $('metrics').textContent='Unable to load evaluation results.'; }
});
if (mode) setupStudio();
function setupStudio() {
  const canvas=$('drawing'),ctx=canvas.getContext('2d',{willReadFrequently:true});
  let drawing=false,hasInk=false,inputMode='draw',selectedFile=null,result=null,selectedIndex=0,history=[],previewUrl=null,busy=false;
  function blank(){ctx.fillStyle='#ffffff';ctx.fillRect(0,0,canvas.width,canvas.height);hasInk=false;$('canvasHint').hidden=false;}
  blank();
  function invalidate(){result=null;$('result').hidden=true;$('emptyResult').hidden=false;$('error').textContent='';}
  function point(e){const b=canvas.getBoundingClientRect();return [(e.clientX-b.left)*canvas.width/b.width,(e.clientY-b.top)*canvas.height/b.height];}
  canvas.addEventListener('pointerdown',e=>{
    if(busy)return;
    history.push({image:ctx.getImageData(0,0,canvas.width,canvas.height),ink:hasInk});if(history.length>20)history.shift();
    invalidate();drawing=true;hasInk=true;$('canvasHint').hidden=true;canvas.setPointerCapture(e.pointerId);
    const [x,y]=point(e);ctx.strokeStyle='#172b20';ctx.fillStyle='#172b20';ctx.lineWidth=Number($('stroke').value);ctx.lineCap='round';ctx.lineJoin='round';ctx.beginPath();ctx.arc(x,y,ctx.lineWidth/2,0,Math.PI*2);ctx.fill();ctx.beginPath();ctx.moveTo(x,y);
  });
  canvas.addEventListener('pointermove',e=>{if(!drawing)return;const [x,y]=point(e);ctx.lineTo(x,y);ctx.stroke();});
  for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>{drawing=false;});
  $('undoButton').onclick=()=>{if(busy)return;const previous=history.pop();if(previous){ctx.putImageData(previous.image,0,0);hasInk=previous.ink;$('canvasHint').hidden=hasInk;invalidate();}};
  $('clearButton').onclick=()=>{if(busy)return;history=[];blank();invalidate();};
  function switchInput(next){if(busy)return;inputMode=next;$('drawPanel').hidden=next!=='draw';$('uploadPanel').hidden=next!=='upload';for(const name of ['draw','upload']){const tab=$(name+'Tab');tab.classList.toggle('selected',name===next);tab.setAttribute('aria-selected',String(name===next));}invalidate();}
  $('drawTab').onclick=()=>switchInput('draw');$('uploadTab').onclick=()=>switchInput('upload');
  function useFile(file){
    if(busy||!file)return;invalidate();
    if(!['image/png','image/jpeg','image/webp'].includes(file.type)||file.size>8*1024*1024){selectedFile=null;$('uploadPreview').hidden=true;$('filename').textContent='';$('error').textContent='Choose a PNG, JPG, or WebP image under 8 MB.';return;}
    selectedFile=file;if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=URL.createObjectURL(file);$('uploadPreview').src=previewUrl;$('uploadPreview').hidden=false;$('filename').textContent=file.name;
  }
  $('fileInput').onchange=e=>useFile(e.target.files[0]);
  $('dropzone').ondragover=e=>e.preventDefault();$('dropzone').ondrop=e=>{e.preventDefault();useFile(e.dataTransfer.files[0]);};
  $('recognizeButton').onclick=async()=>{
    if(busy)return;
    $('error').textContent='';
    if(inputMode==='draw'&&!hasInk){$('error').textContent='Draw something first.';return;}
    if(inputMode==='upload'&&!selectedFile){$('error').textContent='Choose an image first.';return;}
    invalidate();busy=true;$('recognizeButton').disabled=true;$('recognizeButton').textContent='Reading your handwriting…';
    try{
      const blob=inputMode==='draw'?await new Promise(resolve=>canvas.toBlob(resolve,'image/png')):selectedFile;
      const form=new FormData();form.append('mode',mode);form.append('expected',$('expected').value);form.append('image',blob,'handwriting.png');
      const response=await fetch('/api/predict',{method:'POST',headers:{'X-CSRF-Token':csrf},body:form});const data=await response.json();if(!response.ok)throw new Error(data.error||'Unable to read this image.');
      result=data;selectedIndex=0;renderResult();
    }catch(error){$('error').textContent=error.message;}
    finally{busy=false;$('recognizeButton').disabled=false;$('recognizeButton').textContent='Read my handwriting ↗';}
  };
  function renderResult(){
    $('emptyResult').hidden=true;$('result').hidden=false;$('answer').textContent=result.answer;
    $('countLabel').textContent=result.count+' CHARACTER'+(result.count===1?'':'S')+' DETECTED';
    $('correctText').value=result.answer;$('correction').hidden=true;$('feedbackMessage').textContent='';$('yesButton').disabled=false;$('confirmButton').disabled=false;
    $('characterTabs').replaceChildren();
    result.characters.forEach((entry,index)=>{const button=document.createElement('button');button.textContent=entry.answer;button.setAttribute('aria-label','Character '+(index+1)+': '+entry.answer);button.onclick=()=>{selectedIndex=index;renderGuesses();};$('characterTabs').append(button);});renderGuesses();
  }
  function renderGuesses(){
    [...$('characterTabs').children].forEach((button,index)=>button.classList.toggle('selected',index===selectedIndex));
    $('selectedCharacter').textContent='Character '+(selectedIndex+1);$('guesses').replaceChildren();
    result.characters[selectedIndex].guesses.forEach((guess,index)=>{
      const button=document.createElement('button');button.className='guess';button.style.setProperty('--score',guess.confidence+'%');
      const char=document.createElement('b');char.textContent=guess.character;
      if(index===0){const badge=document.createElement('small');badge.textContent='PREDICTED';char.append(badge);}
      const score=document.createElement('span');score.textContent=guess.confidence.toFixed(1)+'%';button.append(char,score);
      button.onclick=()=>{const chars=Array.from($('correctText').value);chars[selectedIndex]=guess.character;$('correctText').value=chars.join('');$('correction').hidden=false;$('correctText').focus();};$('guesses').append(button);
    });
  }
  $('noButton').onclick=()=>{$('correction').hidden=false;$('correctText').focus();};
  $('yesButton').onclick=()=>sendFeedback(result.answer);
  $('confirmButton').onclick=()=>sendFeedback($('correctText').value.trim());
  async function sendFeedback(answer){
    if(!result)return;
    const pattern=mode==='digits'?/^[0-9]+$/:/^[A-Za-z]+$/;
    if(!pattern.test(answer)||answer.length!==result.count){$('feedbackMessage').textContent='Enter exactly '+result.count+' '+(mode==='digits'?'digits':'letters')+'. If detection split incorrectly, redraw with clearer gaps.';return;}
    $('yesButton').disabled=true;$('confirmButton').disabled=true;
    try{
      const response=await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':csrf},body:JSON.stringify({id:result.id,answer,confirmed:true})});const data=await response.json();if(!response.ok)throw new Error(data.error);$('feedbackMessage').textContent=data.message;
    }catch(error){$('feedbackMessage').textContent=error.message;$('yesButton').disabled=false;$('confirmButton').disabled=false;}
  }
  fetch('/api/status').then(r=>r.json()).then(data=>{if(!data.models[mode]){$('modelNotice').hidden=false;$('modelNotice').textContent='This studio needs its trained '+mode+' model. Run python train.py, then return here. You can already try the drawing and upload controls.';}}).catch(()=>{$('error').textContent='Cannot reach the local server.';});
}
