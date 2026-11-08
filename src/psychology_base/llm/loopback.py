"""Shared HTTP isolation for explicitly pinned local transports."""
import urllib.request
from urllib.parse import urlsplit

from psychology_base.llm.execution import remaining_time


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Pinned local transport rejects redirects')


def open_loopback(request, *, base_url, timeout):
    expected, actual = urlsplit(base_url), urlsplit(request.full_url)
    if (expected.scheme != 'http' or expected.hostname != '127.0.0.1'
            or expected.username or expected.password or expected.path or expected.query or expected.fragment
            or type(expected.port) is not int or not 1024 <= expected.port <= 65535
            or actual.scheme != expected.scheme or actual.netloc != expected.netloc
            or actual.username or actual.password or actual.query or actual.fragment):
        raise ValueError('Pinned local transport requires its exact loopback endpoint')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), RejectRedirects())
    return opener.open(request, timeout=remaining_time(timeout))
