const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {enhanceOptsFrom,readEnhancement}=require('../enhance-contract');
const {validateWrestling}=require('../vlm_ocr');

test('invalid sliders fail instead of silently becoming defaults',()=>{
  for(const query of [{scale:'NaN'},{sharpen:'-1'},{contrast:'Infinity'},{denoise:['0','1']},{autoRotate:'maybe'}]) {
    assert.throws(()=>enhanceOptsFrom(query));
  }
  assert.equal(enhanceOptsFrom({scale:'3',conservative:'false'}).scale,3);
  assert.equal(enhanceOptsFrom({conservative:'false'}).conservative,false);
});
test('worker errors and malformed output cannot produce success',()=>{
  for(const out of ['garbage','null','{}','{"error":"cannot read image"}']) assert.throws(()=>readEnhancement(out,'missing'));
});
test('worker result needs an actual PNG, not merely an existing file',()=>{
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'enhance-contract-'));
  const file=path.join(dir,'out.png');
  const result=JSON.stringify({output_size:'100x200',yolo_cropped:false});
  try {
    assert.throws(()=>readEnhancement(result,file));
    fs.writeFileSync(file,'not an image');assert.throws(()=>readEnhancement(result,file));
  }finally{fs.rmSync(dir,{recursive:true});}
});
test('category doubt is flagged for review, never silently accepted or dropped',()=>{
  for(const meta of [null,[],{parse_error:true}]) assert.ok(validateWrestling(meta).error);
  for(const meta of [{player_name:'Unknown'},{category:'baseball'},{category:'wrestling',promotion:'NBA'},{error:'unsupported_category'}]){
    const v=validateWrestling(meta);assert.equal(v.error,undefined);assert.equal(v.review_needed,true);
    assert.ok(v.warnings.some(w=>w.startsWith('category_')),JSON.stringify(v));
  }
  const good=validateWrestling({category:'wrestling',promotion:'AEW',player_name:'Kenny Omega'});
  assert.equal(good.player_name,'Kenny Omega');assert.equal(good.review_needed,true);assert.deepEqual(good.warnings,[]);
  assert.equal(validateWrestling({error:'OpenAI 429: rate'}).error,'OpenAI 429: rate');
});

test('a card with no promotion and no wrestling-specific field is flagged, even if category says wrestling',()=>{
  const v=validateWrestling({category:'wrestling',player_name:'Fred Dryer',manufacturer:'Leaf',set_name:'Pop Century'});
  assert.equal(v.error,undefined);
  assert.ok(v.warnings.includes('category_not_confirmed_wrestling'),JSON.stringify(v));
});
test('promotion alone, or any one wrestling-specific field, is enough to pass',()=>{
  assert.deepEqual(validateWrestling({category:'wrestling',promotion:'AEW',player_name:'X'}).warnings,[]);
  assert.deepEqual(validateWrestling({category:'wrestling',tag_team:'The Hardys',player_name:'X'}).warnings,[]);
  assert.deepEqual(validateWrestling({category:'wrestling',championship:'AEW World Title',player_name:'X'}).warnings,[]);
});
