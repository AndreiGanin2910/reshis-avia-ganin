"""Решись Авиа. Автор проекта: Ганин Андрей Павлович."""
from datetime import date
from pathlib import Path
import math
import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from auth import router as auth_router, authorized_calculation, store
from sqlalchemy import select

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
QUANTILES = [f'{i / 10:.1f}' for i in range(1, 10)]
app = FastAPI(title='Решись Авиа', version='2.0.0')
app.include_router(auth_router)


@app.middleware('http')
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'same-origin'
    response.headers['X-Frame-Options'] = 'DENY'
    if request.url.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response


@app.get('/healthz')
def health():
    with store.engine.connect() as connection:
        connection.execute(select(1))
    return {'status': 'ok', 'version': '2.0.0'}


def load_data():
    routes = pd.read_csv(DATA / 'routes.csv')
    configured = os.getenv('FORECAST_PATH')
    path = Path(configured).expanduser().resolve() if configured else DATA / 'my_forecast.csv'
    if not configured and not path.exists():
        path = DATA / 'seasonal_forecast.csv'
    forecast = pd.read_csv(path)
    required_r = {'route', 'name', 'base_price', 'price_sensitivity', 'price_min', 'price_max'}
    required_f = {'route', 'date', 'seats'} | {f'{q}_base_demand' for q in QUANTILES}
    if not required_r.issubset(routes.columns) or not required_f.issubset(forecast.columns):
        raise ValueError('В CSV отсутствуют обязательные столбцы.')
    if routes[list(required_r)].isna().any().any() or forecast[list(required_f)].isna().any().any():
        raise ValueError('В обязательных столбцах CSV есть пустые значения.')
    forecast['date'] = pd.to_datetime(forecast['date'], errors='raise').dt.strftime('%Y-%m-%d')
    if routes['route'].duplicated().any() or forecast.duplicated(['route', 'date']).any():
        raise ValueError('Повторяющиеся маршруты или пары маршрут-дата.')
    if not set(forecast['route']).issubset(set(routes['route'])):
        raise ValueError('Маршрут прогноза отсутствует в справочнике.')
    for frame, columns in [(routes, ['base_price','price_sensitivity','price_min','price_max']),
                           (forecast, ['seats'] + [f'{q}_base_demand' for q in QUANTILES])]:
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors='raise')
            if not frame[column].map(math.isfinite).all():
                raise ValueError(f'Некорректные числа в {column}.')
    if (forecast.seats <= 0).any() or (forecast.seats % 1 != 0).any():
        raise ValueError('Число мест должно быть положительным целым.')
    if (routes.price_sensitivity <= 0).any() or (routes.price_min < 0).any() or (routes.price_min > routes.price_max).any():
        raise ValueError('Некорректные параметры цены.')
    values = forecast[[f'{q}_base_demand' for q in QUANTILES]].to_numpy()
    if (values[:, 1:] < values[:, :-1]).any():
        raise ValueError('Квантили пересекаются. Проверьте прогноз перед использованием.')
    return routes.set_index('route'), forecast.set_index(['route','date']), path


ROUTES, FORECAST, FORECAST_FILE = load_data()


class PredictionRequest(BaseModel):
    origin: str = Field(min_length=3, max_length=3)
    destination: str = Field(min_length=3, max_length=3)
    date: date
    price: float = Field(ge=0, allow_inf_nan=False)


def calculate(flight, info, price):
    seats = int(flight['seats'])
    delta = float(info['price_sensitivity']) * (price - float(info['base_price']))
    scenarios = []
    for q in QUANTILES:
        demand = float(flight[f'{q}_base_demand']) - delta
        scenarios.append({'quantile': float(q), 'demand': demand,
                          'sales': min(seats, max(0., demand))})
    median = scenarios[4]
    return {'seats': seats, 'demand': median['demand'], 'sales': median['sales'],
            'remaining_after_forecast': seats - median['sales'], 'quantiles': scenarios}


@app.get('/')
def index():
    return FileResponse(ROOT / 'index.html')


@app.get('/api/options')
def options():
    items = []
    for code, row in ROUTES.iterrows():
        origin, destination = code.split('-', 1)
        dates = sorted(FORECAST.loc[code].index.tolist()) if code in FORECAST.index.get_level_values(0) else []
        items.append({'route': code, 'name': row['name'].strip(), 'origin': origin,
                      'destination': destination, 'dates': dates,
                      'base_price': float(row['base_price']), 'price_min': float(row['price_min']),
                      'price_max': float(row['price_max'])})
    return {'routes': items, 'forecast_source': FORECAST_FILE.name,
            'demo': FORECAST_FILE.name == 'seasonal_forecast.csv'}


@app.post('/api/predict')
def predict(request: PredictionRequest, user=Depends(authorized_calculation)):
    route = request.origin.upper() + '-' + request.destination.upper()
    day = request.date.isoformat()
    if route not in ROUTES.index:
        raise HTTPException(404, 'Маршрут не найден.')
    if (route, day) not in FORECAST.index:
        raise HTTPException(404, 'На эту дату прогноза нет.')
    info = ROUTES.loc[route]
    low, high = float(info.price_min), float(info.price_max)
    if not low <= request.price <= high:
        raise HTTPException(422, f'Допустимая цена: {low:g}–{high:g} ₽.')
    return {'route': route, 'date': day, 'price': request.price,
            **calculate(FORECAST.loc[(route, day)], info, request.price)}
