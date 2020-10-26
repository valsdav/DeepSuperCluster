import pickle
import sys 


sc = pickle.load(open(sys.argv[1], "rb"))

print("Mean: ")
print(sc.mean_)

print("Scale: ")
print(sc.scale_)