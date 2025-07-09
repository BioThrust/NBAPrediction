"""
Basketball Reference Request Utilities

This module provides utility functions for making HTTP requests and Selenium-based web scraping
to Basketball Reference. It includes rate limiting and error handling.
"""

from requests import get
from time import sleep, time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import warnings
from bs4 import BeautifulSoup

# Suppress Selenium warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global variables for rate limiting
last_request = time()
_driver = None  # Lazy initialization


def _get_driver():
    """
    Get or create ChromeDriver instance with lazy initialization.
    
    Returns:
        webdriver.Chrome: Configured ChromeDriver instance
    """
    global _driver
    if _driver is None:
        options = Options()
        options.add_argument('--headless=new')
        # Suppress Selenium/ChromeDriver logging
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        options.add_argument("--log-level=3")  # Suppress most logs (0=ALL, 3=FATAL)
        options.add_argument("--silent")
        options.add_argument("--disable-logging")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        
        # Add more browser-like options to avoid detection
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-plugins")
        options.add_argument("--disable-images")
        options.add_argument("--disable-javascript")
        options.add_argument("--disable-animations")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        _driver = webdriver.Chrome(options=options)
        
        # Execute script to remove webdriver property
        _driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
    return _driver


def get_selenium_wrapper(url, xpath):
    """
    Get data from a webpage using Selenium with rate limiting.
    
    Args:
        url (str): URL to scrape
        xpath (str): XPath selector for the target element
    
    Returns:
        str: HTML table content or None if error
    """
    global last_request
    
    # Verify last request was 5 seconds ago
    if 0 < time() - last_request < 5:
        sleep(5)
    last_request = time()
    
    try:
        driver = _get_driver()
        print(f"Loading page: {url}")
        driver.get(url)
        
        # Wait for page to load
        sleep(3)
        
        # Try to find the element
        try:
            element = driver.find_element(By.XPATH, xpath)
            table_html = f'<table>{element.get_attribute("innerHTML")}</table>'
            print(f"Successfully found table element")
            return table_html
        except Exception as e:
            print(f"Could not find element with XPath {xpath}: {e}")
            # Try alternative approach - get the whole page and parse with BeautifulSoup
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            table = soup.find('table', attrs={'id': 'schedule'})
            if table:
                print(f"Found table using BeautifulSoup fallback")
                return str(table)
            else:
                print(f"No table found in page source")
                return None
                
    except Exception as e:
        print(f'Error obtaining data table: {e}')
        return None


def get_wrapper(url):
    """
    Make HTTP request with rate limiting and retry logic.
    
    Args:
        url (str): URL to request
    
    Returns:
        requests.Response: HTTP response object
    """
    global last_request
    
    # More conservative rate limiting - wait 5 seconds between requests
    if 0 < time() - last_request < 5:
        sleep(5)
    last_request = time()
    
    # Add headers to mimic a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        r = get(url, headers=headers, timeout=30)
        
        # Handle different response codes
        if r.status_code == 200:
            return r
        elif r.status_code == 429:
            retry_time = int(r.headers.get("Retry-After", 60))
            print(f'Rate limited! Retrying after {retry_time} sec...')
            sleep(retry_time)
            return get_wrapper(url)  # Retry once
        elif r.status_code == 403:
            print(f'Access forbidden (403) for {url}')
            print('This might be due to blocking by Basketball Reference')
            return r
        elif r.status_code == 404:
            print(f'Page not found (404) for {url}')
            return r
        else:
            print(f'HTTP {r.status_code} error for {url}')
            return r
            
    except Exception as e:
        print(f'Request failed for {url}: {e}')
        # Return a mock response with status code 500
        from requests.models import Response
        mock_response = Response()
        mock_response.status_code = 500
        return mock_response


def close_driver():
    """
    Close the ChromeDriver instance if it exists.
    Call this when you're done with Selenium operations.
    """
    global _driver
    if _driver is not None:
        _driver.quit()
        _driver = None