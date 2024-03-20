import os 
import sys 
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i","--inputdirs", type=str, nargs="+",
                    help="inputfolders", required=True)
parser.add_argument("-o","--outputdir", type=str, help="outputdir",required=True)
parser.add_argument("-n", "--ngroup", type=int, help="Number of files per group", required=True)
parser.add_argument("-c", "--cores", type=int, help="Number of cores", required=True)
args = parser.parse_args()


os.makedirs(args.outputdir, exist_ok=True)

files = []
for inputdir in args.inputdirs:
    files += [inputdir + '/' +f for f in os.listdir(inputdir)  if f != 'log']

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield  i, lst[i:i + n]


def hadd(d):
    print(d)
    output = "{}/output_all{}.root".format(args.outputdir, d[0]+1)
    if os.path.exists(output): return
    os.system("hadd {} {}".format(output, " ".join(d[1])))
    
pool = Pool(args.cores)
pool.map(hadd, chunks(files, args.ngroup))

