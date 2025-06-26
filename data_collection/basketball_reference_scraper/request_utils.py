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
        
        _driver = webdriver.Chrome(options=options)
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
    
    # Verify last request was 3 seconds ago
    if 0 < time() - last_request < 3:
        sleep(3)
    last_request = time()
    
    try:
        driver = _get_driver()
        driver.get(url)
        element = driver.find_element(By.XPATH, xpath)
        return f'<table>{element.get_attribute("innerHTML")}</table>'
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
    
    # Verify last request was 3 seconds ago
    if 0 < time() - last_request < 3:
        sleep(3)
    last_request = time()
    
    r = get(url)
    while True:
        if r.status_code == 200:
            return r
        elif r.status_code == 429:
            retry_time = int(r.headers["Retry-After"])
            print(f'Retrying after {retry_time} sec...')
            sleep(retry_time)
        else:
            return r


def close_driver():
    """
    Close the ChromeDriver instance if it exists.
    Call this when you're done with Selenium operations.
    """
    global _driver
    if _driver is not None:
        _driver.quit()
        _driver = None