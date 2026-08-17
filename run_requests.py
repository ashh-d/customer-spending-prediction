import urllib.request, json, time

def get_info():
    print('--- GET /info ---')
    try:
        print(urllib.request.urlopen('http://127.0.0.1:5000/info').read().decode())
    except Exception as e:
        print('info error:', e)

def post_predict(payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request('http://127.0.0.1:5000/predict', data=data, headers={'Content-Type':'application/json'})
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        print('payload:', payload)
        print('response:', resp.read().decode())
    except Exception as e:
        print('request error:', e)

if __name__ == '__main__':
    get_info()
    samples = [
        {'no_of_visits':10,'apparel_items':5,'fnv_items':8,'staples_items':12},
        {'no_of_visits':5,'apparel_items':2,'fnv_items':1,'staples_items':3},
        {'no_of_visits':0,'apparel_items':0,'fnv_items':0,'staples_items':0},
    ]
    for s in samples:
        post_predict(s)
        time.sleep(0.2)
