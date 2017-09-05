from bs4 import BeautifulSoup as soup

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
            risk_factors.append(elem.text)
        if '1B' in elem.text:
            flag = 0
            break

     content = ''
    for rf in risk_factors:
        content += rf

    return content
