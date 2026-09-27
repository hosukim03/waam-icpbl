import multiprocessing as mp, sys
def info(_=None):
    import shapely, numpy
    return (sys.executable, sys.prefix, shapely.__version__, shapely.geos_version_string, numpy.__version__)
if __name__ == "__main__":
    print("메인  ", info())
    with mp.Pool(1) as p:
        print("작업자", p.map(info, [0])[0])
