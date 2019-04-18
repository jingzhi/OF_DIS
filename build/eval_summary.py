#!/usr/bin/env python
import sys
import argparse
import os
import re
import csv
import codecs

rx_dict = {
        'flow_name':re.compile(r'(.*)\.flo\n'),
        'error_type':re.compile(r'Using (.*) error measure\n'),
        'ave':re.compile(r'Average: (.*)\n'),
        'std':re.compile(r'Standard deviation: (.*)\n'),
        'r0.0':re.compile(r'R0\.0: (.*)%\n'),
        'r0.5':re.compile(r'R0\.5: (.*)%\n'),
        'r1.0':re.compile(r'R1\.0: (.*)%\n'),
        'r2.0':re.compile(r'R2\.0: (.*)%\n'),
        'r4.0':re.compile(r'R4\.0: (.*)%\n'),
        'r8.0':re.compile(r'R8\.0: (.*)%\n'),
        'a0.50':re.compile(r'A0\.50: (.*)\n'),
        'a0.75':re.compile(r'A0\.75: (.*)\n'),
        'a0.95':re.compile(r'A0\.95: (.*)\n'),
        'valid':re.compile(r'Valid region: (.*)\n'),
        'eof':re.compile(r'EOF\n'),
        
        }

def _parse_line(line):
    for key,rx in rx_dict.items():
        match=rx.search(line)
        if match:
            return key,match
    return None, None
if __name__ == '__main__':
    from sys import argv
    
    resultFile1 = argv[1]
    #resultFile2 = argv[2]
    
    results1 = []
    #results2 = []

    with open(resultFile1,'r') as f:
        line = f.readline()
        while line:
            key,match = _parse_line(line)
            # Identify start of new frame
            if(key == "flow_name"):
                row={}
                row.update({key:match.group(1)})
                line = f.readline()
                key,match = _parse_line(line)
                while key!='eof':
                    if key == 'error_type':
                        row.update({key:match.group(1)})
                    else:
                        row.update({key:float(match.group(1))})
                    line = f.readline()
                    key,match = _parse_line(line)
                results1.append(row)
                line = f.readline()

    #with open(resultFile2,'r') as f:
    #    line = f.readline()
    #    while line:
    #        key,match = _parse_line(line)
    #        # Identify start of new frame
    #        if(key == "flow_name"):
    #            row={}
    #            row.update({key:match.group(1)})
    #            line = f.readline()
    #            key,match = _parse_line(line)
    #            while key!='eof':
    #                if key == 'error_type':
    #                    row.update({key:match.group(1)})
    #                else:
    #                    row.update({key:float(match.group(1))})
    #                line = f.readline()
    #                key,match = _parse_line(line)
    #            results2.append(row)
    #            line = f.readline()

    #csv_address='compOut.csv'
    #exists = os.path.isfile(csv_address)
    #mode = 'a' if (exists) else 'w'
    #fields=['frame name','method 1','method 2','method 2 - method 1']
    ##with open(csv_address,mode) as f:
    ##    w =csv.writer(f)
    ##    if (not exists):
    ##        w.writerow(fields)
    ##    for i in range(0,len(results1)):
    ##        r05_1=results1[i]["r0.5"]
    ##        r05_2=results2[i]["r0.5"]
    ##        w.writerow([results1[i]["flow_name"],r05_1,r05_2,r05_2-r05_1])
    total_pixel=0
    total_r05=0
    total_r10=0
    total_r20=0
    total_r40=0
    total_r80=0
    total_epe=0
    for i in range (0,len(results1)):
        total_r05  +=results1[i]["r0.5"]*results1[i]["valid"]
        total_r10  +=results1[i]["r1.0"]*results1[i]["valid"]
        total_r20  +=results1[i]["r2.0"]*results1[i]["valid"]
        total_r40  +=results1[i]["r4.0"]*results1[i]["valid"]
        total_r80  +=results1[i]["r8.0"]*results1[i]["valid"]
        total_epe  +=results1[i]["ave"]*results1[i]["valid"]
        total_pixel+=results1[i]["valid"]
    print("R0.5:\t{}\nR1.0:\t{}\nR2.0:\t{}\nR4.0:\t{}\nR8.0:\t{}\nEPE:\t{}\n".format(total_r05/total_pixel,total_r10/total_pixel,total_r20/total_pixel,total_r40/total_pixel,total_r80/total_pixel,total_epe/total_pixel,total_r05/total_pixel))





