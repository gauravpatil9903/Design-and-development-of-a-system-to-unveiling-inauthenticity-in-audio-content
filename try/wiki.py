import wikipedia

subject = "Mahatma Gandhi"
search_results = wikipedia.search(subject)
if search_results:
    page = wikipedia.page(search_results[0])
    print(page.content[:5000])  # Print first 1000 characters
else:
    print("No results found")
