from bs4 import BeautifulSoup


def extract_text_from_html(html_content: str) -> str:
    """
    Extract plain text from HTML content, removing all HTML tags and elements.

    Args:
        html_content (str): HTML string to extract text from

    Returns:
        str: Plain text with HTML elements removed
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Get text and clean up whitespace
    text = soup.get_text()

    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())

    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

    # Drop blank lines
    text = ' '.join(chunk for chunk in chunks if chunk)

    return text