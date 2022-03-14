def test_python_version():
    import sys
    assert sys.version_info >= (3, 7), "Python verison needs to be at least 3.7"

