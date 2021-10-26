"""JaxFit command line utility

Currently this is a stub for playing with roofit
"""
import argparse
import time


def main() -> int:
    parser = argparse.ArgumentParser(prog="jaxfit", description=__doc__)
    parser.add_argument(
        "-w",
        "--workspace",
        metavar="ROOTFILE:WORKSPACE",
        help="Workspace to load",
        required=True,
    )
    args = parser.parse_args()

    import ROOT

    from jaxfit.roofit import RooWorkspace

    infile, wsname = args.workspace.split(":")
    fin = ROOT.TFile.Open(infile)
    w = fin.Get(wsname)
    tic = time.monotonic()
    wsnew = RooWorkspace.from_root(
        [
            w.pdf("model_s"),
            w.data("data_obs"),
        ]
    )
    outfile = infile.replace(".root", ".json.gz")
    wsnew.to_file(outfile)
    toc = time.monotonic()
    print(f"Parsed and saved model to {outfile} in {toc-tic}s")

    return 0


if __name__ == "__main__":
    exit(main())
