const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');const os=require('node:os');const path=require('node:path');
const {execFileSync}=require('node:child_process');
const vlm=require('../vlm_ocr');

test('names absent from the transcription are demoted, not kept',()=>{
  const bad=vlm.groundName({player_name:'Jade Cargill',all_text:'FIRST UD\nUD EXCLUSIVES\nAEW\n100/100'});
  assert.equal(bad.player_name,null);assert.equal(bad.player_name_unverified,'Jade Cargill');
  assert.equal(bad.name_grounded,false);assert.deepEqual(bad.warnings,['name_not_in_read_text']);
  const good=vlm.groundName({player_name:'Alex Windsor',all_text:'FIRST UD / ALEX WINDSOR / UD EXCLUSIVES® / 100/100'});
  assert.equal(good.player_name,'Alex Windsor');assert.equal(good.name_grounded,true);assert.equal(good.player_name_unverified,null);
  assert.equal(vlm.groundName({player_name:'Alex Windsor',all_text:'ALEX RUNS'}).player_name,null);
});

test('trademark and accent marks do not break grounding (real card 0819)',()=>{
  const m=vlm.groundName({player_name:'Mercedes Martinez',all_text:'YOU NOW OWN A COMMEMORATIVE MERCEDES MARTINEZ™ SUPERSTAR LOGO PATCH CARD!'});
  assert.equal(m.name_grounded,true);assert.equal(m.player_name,'Mercedes Martinez');
  assert.equal(vlm.groundName({player_name:'Penta El Zero Miedo',all_text:'PENTA EL ZERO MIEDO®'}).name_grounded,true);
  assert.equal(vlm.groundName({player_name:'Rey Fénix',all_text:'REY FENIX'}).name_grounded,true);
});

test('retry never waits less than the exponential floor, even when told 469ms',()=>{
  const h=v=>({headers:{get:()=>v}});
  const body='Limit 200000, Used 200000, Requested 1565. Please try again in 469ms.';
  assert.equal(vlm.retryDelay(h(null),body,0),2000);
  assert.equal(vlm.retryDelay(h(null),body,3),16000);
  assert.equal(vlm.retryDelay(h('45'),'',0),45250);   // longer server hint wins
  assert.equal(vlm.retryDelay(h(null),'',9),30000);   // capped
});

test('looksMisoriented triggers only on sideways text or ungrounded names',()=>{
  assert.equal(vlm.looksMisoriented({text_top:'up',player_name:'Alex Windsor',all_text:'ALEX WINDSOR'}),false);
  assert.equal(vlm.looksMisoriented({text_top:'right',player_name:'Alex Windsor',all_text:'ALEX WINDSOR'}),true);
  assert.equal(vlm.looksMisoriented({text_top:'up',player_name:'Jade Cargill',all_text:'AEW 100/100'}),true);
  assert.equal(vlm.looksMisoriented({text_top:'up',player_name:null,all_text:'AEW'}),true);
  assert.equal(vlm.looksMisoriented({parse_error:true}),false);
});

function mockOpenAI(dir,replies,seen){
  return async(_u,init)=>{
    const r=replies.shift();const msg=JSON.parse(init.body).messages[0].content;
    const f=path.join(dir,`seen${seen.length}.jpg`);fs.writeFileSync(f,Buffer.from(msg[1].image_url.url.split(',')[1],'base64'));
    seen.push({size:execFileSync('python3',['-c',`import cv2;im=cv2.imread(${JSON.stringify(f)});print(im.shape[1],im.shape[0])`]).toString().trim(),
               pick:msg[0].text.includes('upright_panel')});
    return {ok:r.status===200,status:r.status,headers:{get:()=>null},
      json:async()=>({choices:[{message:{content:JSON.stringify(r.body)}}]}),text:async()=>typeof r.body==='string'?r.body:''};
  };
}
function withKey(fn){return async()=>{const k=process.env.OPENAI_API_KEY,f=global.fetch;process.env.OPENAI_API_KEY='test';
  try{await fn();}finally{global.fetch=f;if(k===undefined)delete process.env.OPENAI_API_KEY;else process.env.OPENAI_API_KEY=k;}};}
function card(dir,w,h){const img=path.join(dir,'card.jpg');execFileSync('python3',['-c',`import cv2,numpy as np;cv2.imwrite(${JSON.stringify(img)},np.zeros((${h},${w},3),np.uint8))`]);return img;}

test('replays card 1206: ungrounded sideways read -> panel pick -> grounded upright read',withKey(async()=>{
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'vlm-'));const seen=[];
  global.fetch=mockOpenAI(dir,[
    {status:200,body:{category:'wrestling',player_name:'Jade Cargill',text_top:'up',all_text:'FIRST UD UD EXCLUSIVES AEW 100/100'}},
    {status:200,body:{upright_panel:4}},
    {status:200,body:{category:'wrestling',player_name:'Alex Windsor',text_top:'up',promotion:'AEW',all_text:'FIRST UD ALEX WINDSOR 100/100'}},
  ],seen);
  try{
    const m=await vlm.vlmRead(card(dir,200,100),{provider:'openai'});
    assert.equal(m.error,undefined,JSON.stringify(m));
    assert.equal(m.player_name,'Alex Windsor');assert.equal(m.name_grounded,true);
    assert.equal(m.read_rotation,270);assert.equal(m.orientation_method,'panel_pick');assert.equal(m.orientation_panel,4);
    assert.deepEqual(seen.map(s=>s.pick),[false,true,false]);
    assert.equal(seen[0].size,'200 100');assert.equal(seen[1].size,'1064 1164');assert.equal(seen[2].size,'100 200');
    assert.equal(fs.readdirSync(os.tmpdir()).filter(f=>f.startsWith(`vlm_${process.pid}_`)).length,0);
  }finally{fs.rmSync(dir,{recursive:true});}
}));

test('upright, grounded card costs exactly one call',withKey(async()=>{
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'vlm-'));const seen=[];
  global.fetch=mockOpenAI(dir,[{status:200,body:{category:'wrestling',player_name:'Kendal Grey',text_top:'up',all_text:'KENDAL GREY TOPPS CHROME'}}],seen);
  try{const m=await vlm.vlmRead(card(dir,120,80),{provider:'openai'});
    assert.equal(seen.length,1);assert.equal(m.read_rotation,0);assert.equal(m.orientation_method,'original');}
  finally{fs.rmSync(dir,{recursive:true});}
}));

test('panel 1 pick keeps the original read; orient=false never picks',withKey(async()=>{
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'vlm-'));let seen=[];
  global.fetch=mockOpenAI(dir,[{status:200,body:{category:'wrestling',player_name:null,text_top:'up',all_text:'AEW'}},{status:200,body:{upright_panel:1}}],seen);
  try{
    let m=await vlm.vlmRead(card(dir,80,120),{provider:'openai'});assert.equal(seen.length,2);assert.equal(m.read_rotation,0);
    seen=[];global.fetch=mockOpenAI(dir,[{status:200,body:{category:'wrestling',text_top:'left',all_text:''}}],seen);
    m=await vlm.vlmRead(card(dir,80,120),{provider:'openai',orient:false});assert.equal(seen.length,1);assert.equal(m.orientation_method,'original');
  }finally{fs.rmSync(dir,{recursive:true});}
}));
