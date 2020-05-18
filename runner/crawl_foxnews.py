from selenium import webdriver
import os
import time

def count_luanvan(luanvan_results):
    results = luanvan_results.find_elements_by_tag_name("article")

    return len(results)

DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)) + "/"
pattern_url = "https://www.foxnews.com/category/person/donald-trump"

options = webdriver.ChromeOptions()

driver = webdriver.Chrome(DIR_PATH + "chromedriver/chromedriver", chrome_options=options)

driver.get(pattern_url)

contain = driver.find_element_by_xpath(
                                '//*[@class="row"]/main/section/div')

number = count_luanvan(contain)

i = 1

while (1):
    print (i)
    try:
        link_article = driver.find_element_by_xpath(
                                        '//*[@class="row"]/main/section/div/article[' + str(i) + ']/div[1]/a')
        is_video = link_article.get_attribute('href')

        if 'video' in is_video:
            pass
        else:
            link_article.click()

            title = driver.find_element_by_xpath(
                                                '//*[@class="headline"]').text

            content = driver.find_element_by_xpath(
                                                '//*[@class="article-body"]').text

            file_name = ''.join(e for e in title if e.isalnum())

            with open(DIR_PATH + "news_fox/foxnews_" + file_name, 'w') as f:
                f.write(title)
                f.write("\n")
                f.write(content)

            driver.execute_script("window.history.go(-1)")
            time.sleep(3)
    except:
        pass

    count_ = driver.find_element_by_xpath(
        '//*[@class="row"]/main/section/div')
    count = count_luanvan(count_)

    if i >= count:
        while i >= count:
            load_more = driver.find_element_by_xpath(
                                                    '//*[@class="row"]/main/section/footer/div/a')
            load_more.click()

            time.sleep(2)
            count_ = driver.find_element_by_xpath(
                '//*[@class="row"]/main/section/div')
            count = count_luanvan(count_)

    i += 1