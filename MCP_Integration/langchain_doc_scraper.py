"""
LangChain Documentation Scraper
Scrapes LangChain documentation for local indexing and RAG system
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import time
import os
import json
from urllib.parse import urljoin, urlparse


class LangChainDocScraper:
    """Scrape LangChain documentation for local indexing"""

    def __init__(self, base_url="https://python.langchain.com/docs"):
        self.base_url = base_url
        self.visited_urls = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational Bot for Local Documentation)'
        })

    def is_valid_doc_url(self, url: str) -> bool:
        """Check if URL is a valid documentation page"""
        parsed = urlparse(url)
        return (
            parsed.netloc in ['python.langchain.com', 'docs.langchain.com'] and
            '/docs/' in parsed.path and
            not any(x in url for x in ['.pdf', '.zip', '.jpg', '.png'])
        )

    def scrape_page(self, url: str) -> Dict:
        """Scrape a single documentation page"""
        try:
            print(f"Scraping: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove navigation, footer, and script elements
            for element in soup.find_all(['nav', 'footer', 'script', 'style']):
                element.decompose()

            # Extract main content
            main_content = (
                soup.find('main') or
                soup.find('article') or
                soup.find(class_='content') or
                soup.find(id='content')
            )

            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)

            # Extract title
            title = soup.find('h1')
            title_text = title.get_text(strip=True) if title else url.split('/')[-1]

            # Extract code blocks
            code_blocks = []
            for code in soup.find_all('code'):
                code_text = code.get_text(strip=True)
                if len(code_text) > 10:  # Only meaningful code blocks
                    code_blocks.append(code_text)

            return {
                'url': url,
                'title': title_text,
                'content': text,
                'code_examples': code_blocks[:5],  # Limit to 5 examples
                'source': 'LangChain Documentation',
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def scrape_multiple_pages(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple pages with rate limiting"""
        documents = []

        for i, url in enumerate(urls):
            if url not in self.visited_urls and self.is_valid_doc_url(url):
                doc = self.scrape_page(url)
                if doc:
                    documents.append(doc)
                    self.visited_urls.add(url)

                # Be respectful - rate limit
                time.sleep(0.5)

                # Progress update
                if (i + 1) % 10 == 0:
                    print(f"Progress: {i+1}/{len(urls)} pages scraped")

        return documents

    def scrape_from_sitemap(self, sitemap_url: str = None) -> List[Dict]:
        """Scrape pages listed in sitemap"""
        if sitemap_url is None:
            # Try common sitemap locations
            possible_sitemaps = [
                f"{self.base_url}/sitemap.xml",
                "https://python.langchain.com/sitemap.xml",
                "https://docs.langchain.com/sitemap.xml"
            ]
        else:
            possible_sitemaps = [sitemap_url]

        urls = []
        for sitemap in possible_sitemaps:
            try:
                print(f"Trying sitemap: {sitemap}")
                response = self.session.get(sitemap, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'xml')
                    urls = [loc.text for loc in soup.find_all('loc')]
                    print(f"✓ Found {len(urls)} URLs in sitemap")
                    break
            except Exception as e:
                print(f"  Sitemap not found: {e}")
                continue

        if not urls:
            print("Warning: No sitemap found, using manual URL list")
            return self.scrape_common_pages()

        # Limit to first 100 pages for demo
        return self.scrape_multiple_pages(urls[:100])

    def scrape_common_pages(self) -> List[Dict]:
        """Scrape common important documentation pages"""
        common_urls = [
            f"{self.base_url}/get_started/introduction",
            f"{self.base_url}/get_started/quickstart",
            f"{self.base_url}/modules/model_io/models/",
            f"{self.base_url}/modules/model_io/prompts/",
            f"{self.base_url}/modules/memory/",
            f"{self.base_url}/modules/chains/",
            f"{self.base_url}/modules/agents/",
            f"{self.base_url}/use_cases/question_answering/",
            f"{self.base_url}/use_cases/chatbots/",
            f"{self.base_url}/integrations/anthropic/",
        ]

        print(f"Scraping {len(common_urls)} common pages...")
        return self.scrape_multiple_pages(common_urls)

    def save_documents(self, documents: List[Dict], output_file: str):
        """Save scraped documents to JSON file"""
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.',
                    exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Saved {len(documents)} documents to {output_file}")
        print(f"  Total size: {os.path.getsize(output_file) / 1024:.2f} KB")

    def load_documents(self, input_file: str) -> List[Dict]:
        """Load previously scraped documents"""
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)


def main():
    """Main scraper execution"""
    print("="*70)
    print("LangChain Documentation Scraper")
    print("="*70)

    scraper = LangChainDocScraper()

    # Try scraping from sitemap first
    documents = scraper.scrape_from_sitemap()

    # If sitemap fails, scrape common pages
    if not documents:
        documents = scraper.scrape_common_pages()

    # Save results
    output_file = "langchain_docs.json"
    scraper.save_documents(documents, output_file)

    # Statistics
    print(f"\n{'='*70}")
    print("Scraping Statistics:")
    print(f"  Total documents: {len(documents)}")
    print(f"  Total URLs visited: {len(scraper.visited_urls)}")
    print(f"  Output file: {output_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
