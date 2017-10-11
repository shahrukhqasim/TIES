import os
import sys

if len(sys.argv) != 2:
    print("Error")
    sys.exit()

path = sys.argv[1]


for i in os.listdir(path):
    if not i.endswith('.pdf'):
        continue
    without_ext = os.path.splitext(i)[0]
    command = "convert -density 300 %s %s-%%01d.png" % (os.path.join(path, i), os.path.join(path, without_ext))
    os.system(command)
    print(command)