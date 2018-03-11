import csv
import re
import numpy as np
import gensim


def sanitize_input(lines):
    """
    Takes reader as input
    Return sanitized reader
    """
    
    for row in lines:
        comment = row[1]
        
        #for now, we're just removing newline characters because they are the 
        #most obvious issue
        sani_comment = re.sub("\\n","",comment)
        
        #trying to get rid of backslash before apostrophes. not working?
        sani_comment = re.sub('\\\\','',sani_comment)
        row[1] = sani_comment
    
    return lines
def get_reader(filename):
    """
    Takes in the name of a csv file in the current directory
    Returns the csv reader object obtained from that file
    """
    filereader = list()
    with open(filename,'rt', encoding='utf-8') as csvfile:
        filereader = csv.reader(csvfile)
        reader_list = list(filereader)
        
    return reader_list
def main():
    filename = "train.csv"

    lines_lst = get_reader(filename)
    sani_lines = sanitize_input(lines_lst)
    
    for row in sani_lines:
        print(row)
    return 0
    
    
main()