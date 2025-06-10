import pickle

# file_path should be sys argv[1] if you want to pass it as a command line argument
import sys
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    print("Usage: python load_pkl.py <path_to_your_file.pkl>")
    sys.exit(1)

# Open the file in binary mode and load the data
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Now 'data' contains the deserialized Python object
print(data)
