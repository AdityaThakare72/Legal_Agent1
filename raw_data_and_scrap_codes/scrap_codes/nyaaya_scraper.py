import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime

def scrape_article(url):
    """
    Scrapes a single article page for its title, content, and FAQs.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title
        title_element = soup.find('h1')
        title = title_element.get_text(strip=True) if title_element else 'No Title Found'

        # Extract main content
        content_element = soup.select_one('div.prose')
        content = content_element.get_text(strip=True) if content_element else ''

        # Extract FAQs
        faq_content = ''
        faq_section = soup.select_one('div.accordion-group')
        if faq_section:
            faqs = []
            for item in faq_section.find_all('div', class_='accordion-block'):
                question_tag = item.select_one('.accordion-block__title p')
                answer_tag = item.select_one('.accordion-block__content')
                if question_tag and answer_tag:
                    question = question_tag.get_text(strip=True)
                    answer = answer_tag.get_text(strip=True)
                    faqs.append(f"Q: {question}\nA: {answer}")
            if faqs:
                faq_content = "\n\n--- FAQs ---\n" + "\n\n".join(faqs)

        full_content = content + faq_content

        # Fallback if no content was found with specific selectors
        if not full_content.strip():
             main_content_area = soup.find('main', id='primary')
             if main_content_area:
                 full_content = main_content_area.get_text(strip=True)
             else:
                 full_content = 'No Content Found'


        return {
            'title': title,
            'url': url,
            'content': full_content,
            'scraped_at': datetime.datetime.now().isoformat()
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def scrape_nyaaya():
    """
    This function scrapes the nyaaya.org website to extract information about Indian laws.
    """
    explainer_url = "https://nyaaya.org/legal-explainers/"
    response = requests.get(explainer_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    category_links = []
    for link in soup.select('div.shadow-card a'):
        if link and link.has_attr('href'):
            category_links.append(link['href'])

    if not category_links:
        print("No category links found.")
        return

    all_article_links = []
    for cat_link in category_links:
        print(f"Scraping category: {cat_link}")
        try:
            response = requests.get(cat_link)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for link in soup.select('div.shadow-card a.text-dark_blue'):
                if link and link.has_attr('href'):
                    all_article_links.append(link['href'])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching category {cat_link}: {e}")


    if not all_article_links:
        print("No article links found across all categories.")
        return

    scraped_data = []
    for link in all_article_links:
        print(f"Scraping: {link}")
        article_data = scrape_article(link)
        if article_data and article_data['content'] != 'No Content Found' and article_data['content'].strip():
            scraped_data.append(article_data)
        else:
            print(f"Failed to scrape content for: {link}")


    # Create a DataFrame and save it to a CSV file
    if scraped_data:
        df = pd.DataFrame(scraped_data)
        df.to_csv('nyaaya_data.csv', index=False)
        print("\nScraping complete. Data saved to nyaaya_data.csv")
    else:
        print("\nNo data was scraped.")


if __name__ == "__main__":
    scrape_nyaaya()
