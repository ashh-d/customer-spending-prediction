import glob
import json
import pandas as pd
from app import create_app


def run_test():
    app = create_app()
    client = app.test_client()
    meta = json.load(open('features.json'))
    features = meta['features']
    csvs = glob.glob('*.csv')
    if not csvs:
        print('no csv')
        return
    df = pd.read_csv(csvs[0])
    row = df.iloc[0]
    payload = {}
    for f in features:
        key = f.lower().replace(' ', '_')
        val = None
        if f in row.index:
            val = row[f]
        elif key in row.index:
            val = row[key]
        else:
            for col in row.index:
                if col.strip().lower() == f.strip().lower():
                    val = row[col]
                    break
        if val is None:
            val = 0
        payload[key] = float(val)

    resp = client.post('/predict', json=payload)
    print('status', resp.status_code)
    print(resp.get_data(as_text=True))


if __name__ == '__main__':
    run_test()
