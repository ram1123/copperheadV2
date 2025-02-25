# read two txt files and get the ratio of 2nd column of each file

file1 = open("mass_resolution_medians.txt", "r")
file2 = open("CalibrationLog.txt", "r")
lines1 = file1.readlines()
lines2 = file2.readlines()
file1.close()
file2.close()

# print(lines1)
# print(lines2)

# get the ratio of 2nd column of each file
ratio = 0
for i in range(len(lines1)):
    line1 = lines1[i].split()
    line2 = lines2[i].split()
    ratio = float(line2[1])/float(line1[1])
    print(f"{i}: {line2[1]} / {line1[1]} = {ratio}")
