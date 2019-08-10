import os
import sys
import subprocess
 
cmd = "./avg-throughput.sh " +  str(sys.argv[1])
download_location = "p100-logs"
res = subprocess.check_output(cmd.split(" "))


for filename in os.listdir(download_location):
    print(filename)

