from bs4 import BeautifulSoup as soup
import numpy as np
import os
#for creating file with given path structure
def create_file(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    f = open(path, 'w')
    f.close()    


def html_to_text(html_file, text_file):
    create_file(text_file)
    with open(html_file) as f:
         data = f.read()
    f.close()     
    html = soup(data, 'html.parser')
    st = html.get_text()
    st_safe = st.encode('ascii','ignore')
    st_safe = st_safe.split('\n')
    
    st_clean = []
    
    for s in st_safe:
        this_s = s.strip() .lower() 
        if not  this_s.isdigit():
           if len(this_s) > 0 :
             if this_s[0] != '\n':
                   st_clean.append(  ( this_s ) + '\n' ) 
                   
    final_st = ''.join(st_clean)
    
    with open(text_file, 'w') as f:
         f.write(final_st) 
    f.close()
    
    

    
    
    
def extract_risk_factor_section(file_name) :
    with open(file_name) as f:
         data = f.read()

    html = soup(data, 'html.parser')

    risk_factors = []
    flag = 0
    for elem in html.find_all('p'):
      if elem.parent.name == 'div':
        if u'Item\xa01A.' in elem.text:
            flag = 1
        if flag:
            #risk_factors.append(elem.text)
            risk_factors.append(elem)
        if '1B' in elem.text:
            flag = 0
            break

    content = ''
    for rf in risk_factors:
        content += rf

#    return content
    return rf