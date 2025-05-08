"""
- This script runs in the fresh terminal having only voms-proxy setup and export
```bash
voms-proxy-init --voms cms --valid 168:00 --out $(pwd)/voms_proxy.txt
export X509_USER_PROXY=$(pwd)/voms_proxy.txt
```
- It reads the file `files_to_delete.txt` which contains the list of files to be deleted.
- Before deleting, it copies the files to a new location given by `outPath`.
"""
import os
import sys
from pathlib import Path

# inFile = Path("files_to_delete.txt")
inFile = Path("files_to_delete_UL2017.txt")
outPath = Path("/depot/cms/hmm/shar1172/LZMAErrorFiles/")

# open the file and read the lines
inFile_text = inFile.read_text()
# print(inFile_text)

for lines in inFile_text.splitlines():
    # print(lines.split("/"))
    tempPath = outPath / lines.split("/")[-3] / lines.split("/")[-2]
    tempPath.mkdir(parents=True, exist_ok=True)
    # print(tempPath)
    temptext = lines.replace("root://eos.cms.rcac.purdue.edu:1094/", "/eos/purdue")
    cmd = f"cp {temptext} {tempPath}/"
    print(cmd)
    os.system(cmd)
    print("===\n")
    cmd = f"gfal-rm {lines}"
    print(cmd)
    os.system(cmd)
    print("=>>>>>>>>>>\n")
