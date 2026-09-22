import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_utils import file_sha, split_images, label_path
from inspect_checkpoint import compare
from ablate import paired_summary
from eval_matrix import per_class_metrics, instance_density

class TrainingToolsTests(unittest.TestCase):
    def test_hash_covers_beyond_64mb(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'large.pt'
            with p.open('wb') as f:f.seek(64*1024*1024);f.write(b'a')
            before=file_sha(p)
            with p.open('r+b') as f:f.seek(64*1024*1024);f.write(b'b')
            self.assertNotEqual(before,file_sha(p))
    def test_absent_is_not_null_or_string_sentinel(self):
        self.assertIn('a',compare({}, {'a':None}, ['a']))
        self.assertIn('a',compare({}, {'a':'<absent>'}, ['a']))
        self.assertEqual(compare({'a':1,'b':2},{'a':1,'b':3},['a']),{})
    def test_no_classes_means_no_measured_ap(self):
        from types import SimpleNamespace
        self.assertEqual(per_class_metrics(SimpleNamespace(ap_class_index=[],maps=[.9,.9]),{0:'card'}),[])
    def test_failed_baseline_is_not_replaced(self):
        rows=[{'variant':'a','seed':0,'value':None},{'variant':'b','seed':0,'value':.9}]
        self.assertEqual(paired_summary(rows,['a','b'])[1]['status'],'insufficient_pairs')
    def test_density_nested_and_backgrounds(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);images=root/'images/val/nested';images.mkdir(parents=True)
            for n in ('a','b','c','d'):(images/(n+'.jpg')).touch()
            lp=label_path(images/'a.jpg');lp.parent.mkdir(parents=True)
            lp.write_text('0 .5 .5 .2 .2\n'*2)
            label_path(images/'b.jpg').write_text('')
            label_path(images/'c.jpg').write_text('0 .5 .5 .2 .2\n'*4)
            with patch('eval_matrix.resolve',return_value={'val':str(root/'images/val')}):
                report=instance_density('unused','val',3)
            self.assertEqual(report['images'],4)
            self.assertEqual(report['median_instances'],1)
            self.assertEqual(report['missing_labels'],1)
            self.assertEqual(report['images_over_max_det'],1)
            listing=root/'val.txt';listing.write_text('./images/val/nested/a.jpg\n')
            self.assertEqual(split_images({'val':str(listing)},'val'),[(images/'a.jpg').resolve()])
    def test_missing_split_does_not_use_val(self):
        with self.assertRaises(ValueError):split_images({'val':'unused'},'test')

if __name__=='__main__':unittest.main()
