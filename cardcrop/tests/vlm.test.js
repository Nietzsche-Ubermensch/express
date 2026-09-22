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
  // partial match is not enough
  assert.equal(vlm.groundName({player_name:'Alex Windsor',all_text:'ALEX RUNS'}).player_name,null);
  // placeholder-style values are caught too
  assert.equal(vlm.groundName({player_name:'Her Finishing Move',all_text:'HARLEY CAMERON'}).player_name,null);
});

test('retry delay honours Retry-After and OpenAI "try again in" text',()=>{
  const h=v=>({headers:{get:()=>v}});
  assert.equal(vlm.retryDelay(h('3'),'',0),3000);
  assert.equal(vlm.retryDelay(h(null),'Please try again in 1.5s.',0),1750);
  assert.equal(vlm.retryDelay(h(null),'try again in 400ms',0),650);
  assert.equal(vlm.retryDelay(h(null),'',2),8000);
});

test('sideways card is rotated upright, 429 is retried, final read is grounded',async()=>{
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'vlm-'));const img=path.join(dir,'card.jpg');
  execFileSync('python3',['-c',`import cv2,numpy as np;cv2.imwrite(${JSON.stringify(img)},np.zeros((100,200,3),np.uint8))`]);
  const saved={fetch:global.fetch,key:process.env.OPENAI_API_KEY};process.env.OPENAI_API_KEY='test';
  const replies=[
    {status:429,body:'Rate limit. Please try again in 5ms.'},
    {status:200,body:{category:'wrestling',player_name:'Jade Cargill',text_top:'right',all_text:'AEW 100/100'}},
    {status:200,body:{category:'wrestling',player_name:'Alex Windsor',text_top:'up',promotion:'AEW',all_text:'FIRST UD ALEX WINDSOR 100/100'}},
  ];const sizes=[];
  global.fetch=async(_u,init)=>{
    const r=replies.shift();
    const url=JSON.parse(init.body).messages[0].content[1].image_url.url;
    const buf=Buffer.from(url.split(',')[1],'base64');const f=path.join(dir,'seen.jpg');fs.writeFileSync(f,buf);
    sizes.push(execFileSync('python3',['-c',`import cv2;im=cv2.imread(${JSON.stringify(f)});print(im.shape[1],im.shape[0])`]).toString().trim());
    return {ok:r.status===200,status:r.status,headers:{get:()=>null},
      json:async()=>({choices:[{message:{content:JSON.stringify(r.body)}}]}),text:async()=>typeof r.body==='string'?r.body:''};
  };
  try{
    const m=await vlm.vlmRead(img,{provider:'openai'});
    assert.equal(m.error,undefined,JSON.stringify(m));
    assert.equal(m.player_name,'Alex Windsor');assert.equal(m.name_grounded,true);
    assert.equal(m.read_rotation,270);assert.equal(m.orientation_passes,1);
    assert.deepEqual(sizes,['200 100','200 100','100 200']); // retried original, then sent the rotated image
    assert.equal(fs.readdirSync(os.tmpdir()).filter(f=>f.startsWith(`vlm_${process.pid}_`)).length,0); // temp cleaned
  }finally{global.fetch=saved.fetch;if(saved.key===undefined)delete process.env.OPENAI_API_KEY;else process.env.OPENAI_API_KEY=saved.key;fs.rmSync(dir,{recursive:true});}
});

test('orient=false sends only the original',async()=>{
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'vlm-'));const img=path.join(dir,'c.jpg');
  execFileSync('python3',['-c',`import cv2,numpy as np;cv2.imwrite(${JSON.stringify(img)},np.zeros((10,20,3),np.uint8))`]);
  const f0=global.fetch;process.env.OPENAI_API_KEY='test';let calls=0;
  global.fetch=async()=>{calls++;return{ok:true,status:200,headers:{get:()=>null},json:async()=>({choices:[{message:{content:'{"category":"wrestling","text_top":"left","all_text":""}'}}]})};};
  try{const m=await vlm.vlmRead(img,{provider:'openai',orient:false});assert.equal(calls,1);assert.equal(m.read_rotation,0);}
  finally{global.fetch=f0;delete process.env.OPENAI_API_KEY;fs.rmSync(dir,{recursive:true});}
});
