def test_import():
    import hnsw
    assert hnsw is not None

def test_algos_import():
   from hnsw import algos
   assert algos is not None

def test_utils_import():
   from hnsw import utils
   assert utils is not None
