def _importROOT():
    try:
        import ROOT
    except ImportError:
        raise RuntimeError("This code path requires ROOT to be available")
    # TODO: check v6.22 at least
    return ROOT
