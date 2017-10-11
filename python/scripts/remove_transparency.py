import os
import sys

if len(sys.argv) != 2:
    print("Error")
    sys.exit()

path = sys.argv[1]


for i in os.listdir(path):
    if not i.endswith('.png'):
        continue
    full_path = os.path.join(path, i)
    command = "convert %s -background white -alpha remove %s" % (full_path, full_path)
    os.system(command)
    print(command)