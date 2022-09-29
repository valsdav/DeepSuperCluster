import os 
import sys 
from multiprocessing import Pool

inputdir = sys.argv[1]
outputdir = sys.argv[2]
ngroup = int(sys.argv[3])

os.makedirs(outputdir, exist_ok=True)

files = [inputdir + '/' +f for f in os.listdir(inputdir) if f != 'log']

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield  i, lst[i:i + n]


def hadd(d):
    print(d)
    output = "{}/output_all{}.root".format(outputdir, d[0]+1)
    if os.path.exists(output): return
    os.system("hadd {} {}".format(output, " ".join(d[1])))
    
pool = Pool()
pool.map(hadd, chunks(files, ngroup))
