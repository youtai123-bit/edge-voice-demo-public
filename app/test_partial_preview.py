import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import queue
import numpy as np
from partial_preview import PrefixPreview, PreviewJob, PartialPreviewWorker
from transcript_archive import recent_transcripts
from voice_demo_multicandidate import MultiCandidateState, CandidateWorker

class PrefixTests(unittest.TestCase):
    def test_only_captured_samples_preroll_and_bounded_memory(self):
        offers=[]; source=PrefixPreview(lambda lang: 1,offers.append,sample_rate=10,interval_s=1,max_seconds=3)
        source.push(np.arange(10),False,'zh',0)
        source.push(np.arange(10,15),True,'zh',1)
        source.push(np.arange(15,20),True,'zh',2)
        np.testing.assert_array_equal(offers[0].samples,np.arange(5,20,dtype=np.float32))
        source.push(np.arange(20,70),True,'zh',3)
        self.assertLessEqual(source.size,30)
        self.assertLessEqual(len(offers[-1].samples),30)
        self.assertEqual(source.finish(),1)
        self.assertEqual(source.size,0)

    def test_language_boundary_does_not_mix_prefixes(self):
        generation=iter([1,2]);offers=[]
        source=PrefixPreview(lambda lang:next(generation),offers.append,sample_rate=10)
        source.push(np.ones(10),True,'zh',0)
        source.push(np.full(10,2),True,'ja',1)
        source.push(np.full(10,2),True,'ja',2)
        self.assertEqual(offers[-1].generation,2)
        self.assertTrue(np.all(offers[-1].samples==2))

class PreviewConcurrencyTests(unittest.TestCase):
    def test_shared_raw_model_is_skipped_when_final_queue_or_lock_is_busy(self):
        state=MultiCandidateState(language='zh');jobs=queue.Queue();stop=threading.Event()
        worker=CandidateWorker({'raw':object()},None,None,jobs,state,Path('.'),False,stop,
          Path('.'),Path('.'),1,'zh','SenseVoice',Path('unused-language.json'),None)
        job=PreviewJob(state.begin_preview('zh'),'zh',np.zeros(16))
        with patch('voice_demo_multicandidate.decode',return_value=('draft',1)) as decoder:
            self.assertEqual(worker.decode_preview(job),'draft')
            worker.final_busy.set();self.assertIsNone(worker.decode_preview(job));worker.final_busy.clear()
            jobs.put(object());self.assertIsNone(worker.decode_preview(job));jobs.get()
            worker._raw_asr_lock.acquire()
            try:self.assertIsNone(worker.decode_preview(job))
            finally:worker._raw_asr_lock.release()
            self.assertEqual(decoder.call_count,1)
            self.assertEqual(state.partial_decode_count,1)

    def test_new_generation_and_final_reject_late_results(self):
        state=MultiCandidateState(language='zh')
        old=PreviewJob(state.begin_preview('zh'),'zh',np.zeros(16))
        new=PreviewJob(state.begin_preview('zh'),'zh',np.zeros(16))
        state.publish_preview(old,'stale')
        self.assertEqual(state.partial_text,'')
        state.publish_preview(new,'new draft')
        state.finish_preview(old.generation)
        self.assertTrue(state.partial_active)
        state.finish_preview(new.generation)
        state.publish_preview(new,'late')
        self.assertFalse(state.partial_active)
        self.assertEqual(state.partial_text,'')
        self.assertEqual(state.transcript,[])

    def test_latest_pending_wins_and_final_busy_defers(self):
        busy,stop,done=threading.Event(),threading.Event(),threading.Event()
        busy.set();decoded=[]
        def decode(job): decoded.append(job.generation);return str(job.generation)
        worker=PartialPreviewWorker(decode,lambda job,text:done.set(),lambda job:True,busy,stop)
        worker.offer(PreviewJob(1,'zh',np.zeros(16)))
        worker.offer(PreviewJob(2,'zh',np.zeros(16)))
        worker.start();time.sleep(.06)
        self.assertEqual(decoded,[])
        busy.clear();self.assertTrue(done.wait(1));stop.set();worker.join(1)
        self.assertEqual(decoded,[2])

    def test_older_inflight_result_is_not_published(self):
        entered,release,stop=threading.Event(),threading.Event(),threading.Event()
        current=[1];published=[]
        def decode(job): entered.set();release.wait(1);return 'old'
        worker=PartialPreviewWorker(decode,lambda job,text:published.append(text),
          lambda job:job.generation==current[0],threading.Event(),stop)
        worker.start();worker.offer(PreviewJob(1,'zh',np.zeros(16)))
        self.assertTrue(entered.wait(1));current[0]=2;release.set();time.sleep(.06)
        stop.set();worker.join(1);self.assertEqual(published,[])

class ArchiveTests(unittest.TestCase):
    def test_history_tail_restores_without_rewriting_and_uses_unique_archive_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'transcript.jsonl'
            content=''.join(json.dumps({'index':i%3,'text':'test '+str(i)})+'\n' for i in range(20))+'{incomplete'
            path.write_text(content,encoding='utf-8');before=path.read_bytes()
            rows=recent_transcripts(path,limit=5,byte_limit=300)
            self.assertEqual(len(rows),5)
            self.assertEqual(rows[-1]['text'],'test 19')
            self.assertEqual(len({row['record_id'] for row in rows}),5)
            self.assertEqual(path.read_bytes(),before)

    def test_missing_history_is_empty(self):
        self.assertEqual(recent_transcripts(Path('does-not-exist-archive')),[])

if __name__=='__main__':unittest.main()
