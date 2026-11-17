from src.tools.tools import web_search, scrape_url

results = scrape_url.invoke("https://www.nairaland.com/8784486/adc-alleges-plot-dress-thugs")
print("Scraped Content:\n", results)

