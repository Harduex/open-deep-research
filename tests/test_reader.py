from open_deep_research.core.reader import Reader


def test_extract_links_basic():
    html = '''
    <a href="https://example.com/page1">Page 1</a>
    <a href="https://example.com/page2">Page 2</a>
    <a href="/relative">Relative</a>
    '''
    urls = Reader._extract_links(html, "https://base.com")
    assert "https://example.com/page1" in urls
    assert "https://example.com/page2" in urls
    assert "https://base.com/relative" in urls


def test_extract_links_filters_non_http():
    html = '''
    <a href="mailto:test@example.com">Email</a>
    <a href="javascript:void(0)">JS</a>
    <a href="ftp://files.example.com">FTP</a>
    <a href="https://valid.com">Valid</a>
    '''
    urls = Reader._extract_links(html, "https://base.com")
    assert len(urls) == 1
    assert urls[0] == "https://valid.com"


def test_extract_links_removes_fragments():
    html = '<a href="https://example.com/page#section1">Link</a>'
    urls = Reader._extract_links(html, "https://base.com")
    assert urls[0] == "https://example.com/page"


def test_extract_links_deduplicates():
    html = '''
    <a href="https://example.com/page">Link 1</a>
    <a href="https://example.com/page">Link 2</a>
    <a href="https://example.com/page#frag">Link 3</a>
    '''
    urls = Reader._extract_links(html, "https://base.com")
    assert len(urls) == 1


def test_extract_links_caps_at_max():
    links = "\n".join(f'<a href="https://example.com/page{i}">Link</a>' for i in range(50))
    urls = Reader._extract_links(links, "https://base.com")
    assert len(urls) == 20
