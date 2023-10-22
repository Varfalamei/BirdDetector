import os
import time
import datetime
import requests
from bs4 import BeautifulSoup

from loguru import logger
import pandas as pd
from requests.exceptions import RequestException, Timeout


def try_to_find_word(article_text: str) -> bool:
    """
    Check special words from list in a text
    :param article_text: text
    :return: False / True
    """

    lst_with_word = ["mercedes", "мерседес"]
    for word in lst_with_word:
        if word in article_text.lower():
            return True

    return False


def content_from_WIKI_page(url: str) -> dict:
    lst_with_links = []
    try:
        response = requests.get(url, timeout=10)  # Установите желаемый таймаут, например, 10 секунд.
    except (RequestException, Timeout) as e:
        print(f"Error while fetching {url}: {e}")
        return {
            "url": url,
            "status": "Error"
        }

    if response.status_code == 200:
        page_content = response.text
        soup = BeautifulSoup(page_content, 'html.parser')

        # Удаление ссылок на примечание по типу [3], [5] и т.д.
        reference_sup_elements = soup.find_all("sup", {"class": "reference"})
        for sup_element in reference_sup_elements:
            sup_element.decompose()

        # Удаление ссылок на [править| смотреть код]
        edit_section_elements = soup.find_all(class_="mw-editsection")
        for edit_section_element in edit_section_elements:
            edit_section_element.decompose()

        for heading_id in ["См._также", "Примечания", "Библиография", "Ссылки", "Литература", "Галерея"]:
            first_heading = soup.find("span", {"class": "mw-headline", "id": heading_id})
            if first_heading:
                parent_element = first_heading.find_parent()
                for element in parent_element.find_all_next():
                    element.decompose()

        article_text = soup.get_text()

        if try_to_find_word(article_text) is False:
            return {
                "url": url,
                "article_text": article_text.strip(),
                "links": None,
                "status": "No find special words"

            }

        links = soup.find_all('a', href=True)

        for link in links:
            link_url = link['href']
            #  Проверяю, что ссылка введёт на страницу википедии, без изображения флага и картинки машины.
            if (link_url.startswith("/wiki/")
                    and "svg" not in link_url.lower()
                    and "jpg" not in link_url.lower()
                    and "png" not in link_url.lower()
            ):
                full_link = "https://ru.wikipedia.org" + link_url
                lst_with_links.append(full_link)

        return {
            "url": url,
            "article_text": article_text.strip(),
            "links": lst_with_links,
            "status": "Ok"

        }
    else:
        print(f"url {url}, error:{response.status_code}")

        return {
            "url": url,
            "status": "Error"
        }


def main():
    class Config:
        folder = './corpus_22_10/wiki'
        time_to_sleep = 1
        max_len_corpus = 10000
        first_url = "https://ru.wikipedia.org/wiki/Mercedes-Benz"
        size_csv = 500
        only_with_special_words = False

    # Create logger system and folders for save corpus
    date_now = datetime.datetime.now().strftime("%d_%B_%Y_%H_%M")
    logger.add(f"{Config.folder}/info__{date_now}.log")
    if not os.path.exists(Config.folder):
        os.makedirs(Config.folder)

    lst_url = [Config.first_url]
    list_with_sample = []
    check_urls = []

    # Parser loop
    logger.info("Start")
    counter = 0
    while len(lst_url):
        time.sleep(Config.time_to_sleep)

        # Find an url with "merc" and drop the url from queue or just take the first url
        url = None
        for item in lst_url:
            if "merc" in item.lower():
                url = item
                break

        if url is not None:
            lst_url.remove(url)
        else:
            url = lst_url.pop(0)

        logger.info(f"{counter}, count_sample, {len(list_with_sample)}, len queue: {len(lst_url)}, url {url},")

        # Check url in the list of passed urls
        if url in check_urls:
            continue
        check_urls.append(url)

        # Parsing url
        result = content_from_WIKI_page(url=url,
                                        )

        if result["status"] == "Error":
            sample = {
                "url": result['url'],
                "context": None,
                "status": "Error"
            }
        elif result["status"] == "No find special words":
            sample = {
                "url": result['url'],
                "context": result['article_text'],
                "status": "No find special words"
            }
        else:
            sample = {
                "url": result['url'],
                "context": result['article_text'],
                "status": "Ok"
            }
            lst_url.extend(result["links"])

        list_with_sample.append(sample)
        counter += 1

        # Create df
        if len(list_with_sample) == Config.size_csv:
            df = pd.DataFrame(list_with_sample)
            name_df = str(len(os.listdir(Config.folder)))
            df.to_csv(f"{Config.folder}/{name_df}.csv", index=False)
            list_with_sample = []
            logger.info(f"df {name_df}.csv is saved")

        if counter % 5 == 0:

            ## Добавить логику удаляния повторов (сет?)

            logger.info('Check url from check urls in lst_url')
            logger.info(f'Old size: lst_url {len(lst_url)}, check_urls {len(check_urls)}')
            lst_url = list(set(lst_url))
            lst_url = [url for url in lst_url if url not in check_urls]

            lst_url.sort(key=lambda x: "merc" not in x.lower())

            logger.info(f'New size: lst_url {len(lst_url)}, check_urls {len(check_urls)}')

        # Check of number of processed pages
        if counter > Config.max_len_corpus:
            logger.info(f"More than {Config.max_len_corpus} pages have been uploaded. Finished")
            df = pd.DataFrame(list_with_sample)
            name_df = str(len(os.listdir(Config.folder)))
            df.to_csv(f"{Config.folder}/{name_df}.csv", index=False)
            break

    # Saving a list of wasn't processed pages
    with open("no_processed_pages.txt", "w", encoding="utf-8") as file:
        for item in lst_url:
            file.write(item + "\n")

    logger.info("Finished")


if __name__ == '__main__':
    main()
