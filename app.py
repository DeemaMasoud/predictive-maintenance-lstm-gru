
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from xai import explain_shap, explain_ig, make_heatmap

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

lstm = load_model('lstm.keras', compile=False)
gru = load_model('gru.keras', compile=False)
sc = joblib.load('scaler.pkl')

cols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [
    f's{i}' for i in range(1, 22)
]

features = ['op1', 'op2', 's2', 's3', 's4', 's6', 's7',
            's8', 's9', 's11', 's12', 's13', 's14',
            's15', 's17', 's20', 's21']
sensor_names = {
    'op1': 'Operational Setting 1',
    'op2': 'Operational Setting 2',
    's2': 'Low-pressure compressor outlet temperature',
    's3': 'High-pressure compressor outlet temperature',
    's4': 'Low-pressure turbine outlet temperature',
    's6': 'Bypass duct pressure',
    's7': 'High-pressure compressor outlet pressure',
    's8': 'Physical fan speed',
    's9': 'Physical core rotational speed',
    's11': 'High-pressure compressor outlet static pressure',
    's12': 'Fuel flow to compressor pressure ratio',
    's13': 'Corrected fan speed',
    's14': 'Corrected core speed',
    's15': 'Bypass ratio',
    's17': 'Bleed enthalpy',
    's20': 'High-pressure turbine coolant bleed',
    's21': 'Low-pressure turbine coolant bleed'
}



def read_data(file):
    name = file.filename.lower()

    if name.endswith('.csv'):
        return pd.read_csv(file)

    elif name.endswith('.xlsx'):
        return pd.read_excel(file)

    elif name.endswith('.txt'):
        data = pd.read_csv(
            file, sep=r'\s+', header=None
        )

        if data.shape[1] != 26:
            raise ValueError(
                'NASA TXT files must contain 26 columns.'
            )

        data.columns = cols
        return data

    else:
        raise ValueError(
            'Please upload a CSV, TXT, or XLSX file.'
        )


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            file = request.files.get('file')

            if not file or not file.filename:
                raise ValueError('Please choose a file.')

            data = read_data(file)

            if not all(c in data.columns for c in features):
                raise ValueError(
                    'Missing required sensor columns.'
                )

            engine_id = request.form.get('engine_id', '').strip()

            if 'unit' in data.columns:
                data['unit'] = pd.to_numeric(
                    data['unit'], errors='raise'
                )

                engine_count = data['unit'].nunique()

                if engine_id:
                    selected = int(engine_id)
                    data = data[data['unit'] == selected]

                    if data.empty:
                        raise ValueError(
                            f'Engine {selected} was not found.'
                        )

                elif engine_count > 1:
                    raise ValueError(
                        'This file contains multiple engines. '
                        'Please enter an Engine ID.'
                    )

            elif engine_id:
                raise ValueError(
                    'This file has no unit column. '
                    'Leave Engine ID empty.'
                )

            if len(data) < 30:
                raise ValueError(
                    'At least 30 cycles are required.'
                )

            if 'cycle' in data.columns:
                data['cycle'] = pd.to_numeric(
                    data['cycle'], errors='raise'
                )

                if data['cycle'].duplicated().any():
                    raise ValueError(
                        'Duplicate operating cycles found.'
                    )

                data = data.sort_values('cycle')

            values = data[features].tail(30).apply(
                pd.to_numeric, errors='raise'
            )

            if not np.isfinite(values.to_numpy()).all():
                raise ValueError(
                    'Sensor data contains invalid values.'
                )

            scaled = sc.transform(values).astype(np.float32)
            x = np.array([scaled])

            p1 = float(lstm.predict(x, verbose=0)[0][0])
            p2 = float(gru.predict(x, verbose=0)[0][0])

            shap_data = None
            heatmap = None
            xai_error = None

            try:
                shap_data = explain_shap(gru, scaled, features)
                for item in shap_data:
                    name = item['name']
                    readings = values[name].to_numpy(dtype=float)

                    first = readings[:5].mean()
                    last = readings[-5:].mean()

                    change = (last - first) / max(abs(first), 1e-8) * 100

                    if change > 1:
                        trend = f"increased by {change:.2f}%"
                    elif change < -1:
                        trend = f"decreased by {abs(change):.2f}%"
                    else:
                        trend = "remained relatively stable"

                    contribution = abs(item['value'])

                    if item['value'] >= 0:
                        effect = f"increased the predicted RUL by {contribution:.3f} cycles"
                    else:
                        effect = f"decreased the predicted RUL by {contribution:.3f} cycles"

                    item['label'] = sensor_names[name]
                    item['trend'] = (
                        f"The average reading {trend} between the first "
                        f"five and last five cycles."
                    )
                    item['explanation'] = (
                        f"According to SHAP, this feature {effect} "
                        f"relative to the selected baseline."
                    )

                ig = explain_ig(gru, scaled)
                heatmap = make_heatmap(ig, features)

            except Exception as e:
                xai_error = str(e)
            return render_template(
                'index.html',
                lstm=round(p1, 2),
                gru=round(p2, 2),
                shap_data=shap_data,
                heatmap=heatmap,
                xai_error=xai_error
            )

        except Exception as e:
            return render_template(
                'index.html', error=str(e)
            )

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
