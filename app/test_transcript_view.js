const assert=require('node:assert/strict');
const {TranscriptBuffer}=require('./transcript_view.js');
const storage={data:new Map(),getItem(k){return this.data.get(k)||null;},setItem(k,v){this.data.set(k,v);}};
const buffer=new TranscriptBuffer(storage);
const row=(index,text,session='s1',generation=index)=>({index,text,session_id:session,
  record_id:session+':'+index,preview_generation:generation,language:'zh',latency_ms:100});
let state={session_id:'s1',partial_generation:2,partial_active:true,partial_text:'draft',transcript:[row(1,'first')]};
assert.equal(buffer.ingest(state).rows.length,1);
assert.equal(buffer.ingest(state).rows.length,1); // no duplication while polling
const before=JSON.stringify(state);
buffer.clear(state);
assert.equal(JSON.stringify(state),before); // display-only: no backend mutation
assert.deepEqual(buffer.ingest(state),{rows:[],draft:null});
let refreshed=new TranscriptBuffer(storage);
assert.equal(refreshed.ingest(state).rows.length,0); // cleared text stays cleared after refresh
state={...state,partial_active:false,transcript:[row(1,'first'),row(2,'cleared while drafting')]};
assert.equal(refreshed.ingest(state).rows.length,0); // late final of cleared draft stays hidden
state={...state,partial_generation:3,transcript:[...state.transcript,row(3,'new sentence')]};
assert.equal(refreshed.ingest(state).rows[0].text,'new sentence');
state.transcript[2].text='corrected';
assert.equal(refreshed.ingest(state).rows[0].text,'corrected');
const restarted={session_id:'s2',partial_generation:0,partial_active:false,transcript:state.transcript};
assert.equal(refreshed.ingest(restarted).rows.length,1); // restored archive rows retain stable IDs
refreshed.clear(restarted);
assert.equal(new TranscriptBuffer(storage).ingest(restarted).rows.length,0);
const fresh={session_id:'s3',partial_generation:1,partial_active:false,transcript:[row(1,'after restart','s3')]};
assert.equal(refreshed.ingest(fresh).rows[0].text,'after restart');
console.log('PASS: append, dedupe, correction update, refresh persistence, display-only clear, late draft/final, stable archive IDs and restart.');
