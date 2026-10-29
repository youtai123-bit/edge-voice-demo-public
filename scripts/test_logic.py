"""Only pure logic tests; no models, audio hardware, network or inference."""
from pathlib import Path
import sys, unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
suite=unittest.TestLoader().loadTestsFromNames([
  'test_candidate_selection','test_language_selection',
  'test_personalization','test_asr_backends'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
