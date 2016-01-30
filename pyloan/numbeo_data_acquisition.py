from tqdm import tqdm
import json, requests, re


country_mapping = {'Argentina':'AR', 'Australia':'AU', 'Austria':'AT', 'Bangladesh':'BD', 'Belgium':'BE',
                   'Burkina Faso':'BF', 'Bulgaria':'BG', 'Bahrain':'BH', 'Burundi':'BI', 'Benin':'BJ',
                   'Saint Barthelemy':'BL', 'Bermuda':'BM', 'Brunei Darussalam':'BN', 'Bolivia':'BO','Bonaire':'BQ',
                   'Brazil':'BR', 'Canada':'CA',"Cote d'Ivoire":'CI', 'Chile':'CL', 'China':'CN', 'Colombia':'CO',
                   'Czech Republic':'CZ', 'Germany':'DE', 'Denmark':'DK', 'Dominican Republic':'DO', 'Algeria':'DZ',
                   'Ecuador':'EC', 'Estonia':'EE', 'Egypt':'EG', 'Finland':'FI', 'France':'FR', 'Ghana':'GH',
                   'Greece':'GR', 'Guyana':'GY', 'Croatia':'HR', 'Hungary':'HU', 'India':'IN', 'Indonesia':'ID',
                   'Ireland':'IE', 'Israel':'IL', 'Iran':'IR', 'Iceland':'IS', 'Italy':'IT', 'Jamaica':'JM',
                   'Japan':'JP', 'Kenya':'KE', 'Lebanon':'LB', 'Sri Lanka':'LK', 'Lithuania':'LT', 'Luxembourg':'LU',
                   'Latvia':'LV', 'Morocco':'MA', 'Macedonia':'MK', 'Mauritius':'MU', 'Mexico':'MX', 'Namibia':'NA',
                   'Netherlands':'NL', 'Nigeria':'NG', 'Norway':'NO', 'New Zealand':'NZ', 'ON':'CA', 'Panama':'PA',
                   'Peru':'PE', 'Philippines':'PH', 'Pakistan':'PK', 'Poland':'PL', 'Portugal':'PT', 'Romania':'RO',
                   'Serbia':'RS', 'Russian Federation':'RU', 'Singapore':'SG', 'Slovenia':'SI', 'Slovakia':'SK',
                   'Suriname':'SR', 'Syrian Arab Republic':'SY', 'Thailand':'TH', 'Tanzania':'TZ', 'Ukraine':'UA',
                   'South Africa':'ZA', 'South Korea':'KR', 'Spain':'ES', 'Sweden':'SE', 'Switzerland':'CH',
                   'Viet Nam':'VN', 'Zimbabwe':'ZW', 'Turkey':'TR', 'United Kingdom':'GB', 'United States':'US',
                   'Venezuela':'VE'
                  }

def map_country_iso(country):
    if country not in country_mapping.keys():
        return 'PROBLEM WITH : %s' % country
    else:
        return country_mapping.get(country)

regex1 = re.compile("u\'|\'")
searching_for_average = re.compile('Average Monthly Disposable Salary \(After Tax\), Salaries And Financing')

def average_salary_extraction(prices): 
    prices.replace(regex1, '\"')#.apply(lambda price: json.loads(price))
    prices = prices.apply(lambda price: [dictio['average_price'] for dictio in price
                                 if searching_for_average.match(dictio['item_name'])])
    return prices.apply(lambda price: price[0])

def get_numbeo_data_according_to_location(quadruple_id_lat_long_region):    
    all_city_prices_requests = []
    all_country_prices_requests = []
    missing_prices_location = []
    for address in tqdm(quadruple_id_lat_long_region):
        # CITY REQUEST
        url = "http://www.numbeo.com/api/city_prices?api_key=w624aixu3sour8&query="+str(address[1])+','+str(address[2])
        response = json.loads(requests.get(url.strip()).text)
        # CHECK CITY REQUEST RESPONSE
        if 'error' in response.keys():
            # COUNTRY REQUEST BECAUSE CITY REQUEST FAILED
            url = "http://www.numbeo.com/api/country_prices?api_key=w624aixu3sour8&country="+address[3]
            response = json.loads(requests.get(url.strip()).text)
            # CHECK COUNTRY RESPONSE
            if 'error' in response.keys():
                print "ERROR CITY-COUNTRY"
                print address
                missing_prices_location.append(address)
                continue
            else:
                # CHECK ITEM OF LIST
                if 'prices' in response.keys() and isinstance(response.get('prices'), list):
                    boolean = False
                    for dico in response.get('prices'):
                        if 105 in dico.values():
                            response['loan_identifier'] = address[0]
                            response['region'] = address[3]
                            all_country_prices_requests.append(response)
                            boolean = True
                            break
                    if not boolean:
                        print "ERROR CITY + NOT ALL PRICE PRESENT IN COUNTRY REQUEST"
                        print address
                        missing_prices_location.append(address)
                        continue
        else:
            if 'prices' in response.keys() and isinstance(response.get('prices'), list):
                boolean = False
                for dico in response.get('prices'):
                    if 105 in dico.values():
                        response['loan_identifier'] = address[0]
                        response['address_lat'] = address[1]
                        response['address_lng'] = address[2]
                        all_city_prices_requests.append(response)
                        boolean = True
                        break
                if not boolean:
                    country = response['name'].split(', ')
                    if len(country)==3 and country[2]=="United States":
                        country = country[2]
                    else:
                        country = country[1]
                    country = map_country_iso(country)
                    url = "http://www.numbeo.com/api/country_prices?api_key=w624aixu3sour8&country="+country
                    response = json.loads(requests.get(url).text)
                    if 'error' in response.keys() or 'prices' not in response.keys():
                        print "ERROR COUNTRY - CITY NOT FOUND"
                        print country
                        print address
                        missing_prices_location.append(address)
                        continue
    print "Done downloading from numbeo website!"
    return (all_city_prices_requests, all_country_prices_requests, missing_prices_location)

def get_numbeo_data_according_to_region(regions):
    regions_prices_requests = []
    missing_prices_regions = []
    for region in tqdm(regions):
        url = "http://www.numbeo.com/api/country_prices?api_key=w624aixu3sour8&country="+region
        response = json.loads(requests.get(url).text)
        if 'error' in response.keys():
            missing_prices_regions.append(region)
        else:
            if 'prices' in response.keys() and isinstance(response.get('prices'), list):
                boolean = False
                for dico in response.get('prices'):
                    if 105 in dico.values():
                        response['region'] = region
                        regions_prices_requests.append(response)
                        boolean = True
                        break
                if not boolean:
                    print "ERROR CITY + NOT ALL PRICE PRESENT IN COUNTRY REQUEST"
                    print region
                    missing_prices_regions.append(region)
                    continue
    print "Done downloading from numbeo website!"
    return (regions_prices_requests, missing_prices_regions)