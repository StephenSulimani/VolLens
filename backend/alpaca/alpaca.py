import requests


class Alpaca:
    def __init__(self, api_key, api_secret):
        self.authorization_headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }
        self.session = requests.Session()
        self.session.headers.update(self.authorization_headers)

        self.base_url = "https://data.alpaca.markets/v1beta1/"

    def _send_request(self, method: str, endpoint: str, **kwargs):
        url = self.base_url + endpoint
        self.session.headers.update(self.authorization_headers)
        return self.session.request(method, url, **kwargs)
