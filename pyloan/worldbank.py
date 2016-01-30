"""
API for the worldbank
http://data.worldbank.org/developers/api-overview

"""
import requests

import pandas as pd

from bitbond.data.load import loan_data
from bitbond.data.constants import WORLDBANK_DATA_PATH


def get_country_gnp_per_capita(country_iso):
    """ Gets country GNP per capita in current $ values

    :param country_iso:
    :type country_iso:
    :param year:
    :type year:
    :return:
    :rtype: dict
    """
    url = 'http://api.worldbank.org/countries/%s/indicators/NY.GNP.PCAP.CD/' % country_iso
    # Query some more years, in case we lack data
    date_str = '2000:2020'
    r = requests.get(url=url, params={'format': 'json', 'date': date_str})
    if len(r.json()) == 1:
        return {}
    header, records = r.json()
    result = {rec['date']: rec['value'] for rec in records}
    return result


def crawl_gnp_per_capita(countries):
    return {country: get_country_gnp_per_capita(country) for country in countries}


if __name__ == '__main__':
    data = loan_data()
    countries = set(data['region'])
    gnp_data = crawl_gnp_per_capita(countries)
    gnp_df = pd.DataFrame.from_dict(gnp_data)
    gnp_df.to_csv(WORLDBANK_DATA_PATH)
