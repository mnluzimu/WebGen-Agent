from selenium import webdriver, __version__ as sel_ver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

print("Selenium", sel_ver)
opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--no-sandbox")
opts.add_argument("--disable-dev-shm-usage")
opts.add_argument("--window-size=1280,800")

driver = webdriver.Chrome(service=Service(), options=opts)
driver.get("https://example.com")
print("Title =", driver.title)
driver.quit()
