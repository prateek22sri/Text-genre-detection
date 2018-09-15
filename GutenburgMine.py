"""
This file mines the data from the Gutenburg.org which offers over 54,000 free books
"""

from selenium import webdriver
import urllib.request

driver = webdriver.Chrome()
driver.get("https://www.gutenberg.org/wiki/Category:Religion_Bookshelf")

driver2 = webdriver.Chrome()


driver3 = webdriver.Chrome()



d = driver.find_elements_by_tag_name("a")

num = 1
for i in d:
    if num > 1000:
        break
    k = i.get_attribute('href')
    if k:
        if "Book" in k:
            driver3.get(k)
            kk = driver3.find_elements_by_tag_name("a")

            for l in kk:

                s = l.get_attribute('href')

                if s:
                    if 'ebooks' in s:
                        driver2.get(s)
                        p = driver2.find_elements_by_tag_name("a")

                        f = open("/home/hduser/ANLP_project/data/Religion/"+str(num), "wb")
                        for j in p:
                            t = j.get_attribute('text')


                            if 'Plain' in t:
                                book = j.get_attribute('href')

                                data = urllib.request.urlopen(book)  # it's a file like object and works just like a file
                                for line in data:
                                    f.write(line)
                        num+=1
                        if num>1000:
                            break
                        f.close()


driver.close()
driver2.close()
driver3.close()




