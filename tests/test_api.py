import sys
import unittest
import tempfile
import os
from pathlib import Path
TEST_STATE = tempfile.TemporaryDirectory()
os.environ['DATABASE_URL'] = 'sqlite:///' + str(Path(TEST_STATE.name) / 'test.sqlite3')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from app import app, calculate, ROUTES, FORECAST
from auth import store, RateLimiter

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        store.limiter = RateLimiter()
        cls.client = TestClient(app)
        response = cls.client.post('/api/auth/register', json={
            'username':'api_test_user','display_name':'API test','password':'testing-only-password-123'})
        assert response.status_code == 201, response.text
        cls.client.headers['X-CSRF-Token'] = response.json()['csrf_token']
    def request(self, **changes):
        data = dict(origin='MOW', destination='AER', date='2026-06-12', price=10000)
        return self.client.post('/api/predict', json={**data, **changes})
    def test_reference_fixture(self):
        fixture = {'seats':900, **{f'{i/10:.1f}_base_demand':970 for i in range(1,10)}}
        info = {'base_price':9000, 'price_sensitivity':0.141}
        for price, demand, sales in [(9000,970,900),(10000,829,829),(18000,-299,0)]:
            result = calculate(fixture, info, price)
            self.assertAlmostEqual(result['demand'], demand)
            self.assertAlmostEqual(result['sales'], sales)
            self.assertEqual(result['quantiles'][4]['demand'], result['demand'])
    def test_api_contract(self):
        r = self.request()
        self.assertEqual(r.status_code,200)
        d = r.json()
        self.assertEqual(len(d['quantiles']),9)
        self.assertLessEqual(d['sales'],d['seats'])
        self.assertGreaterEqual(d['sales'],0)
    def test_invalid_inputs(self):
        for change, status in [({'destination':'XXX'},404),({'date':'2028-01-01'},404),
                               ({'date':'bad'},422),({'price':-1},422),({'price':1},422)]:
            self.assertEqual(self.request(**change).status_code,status)
    def test_all_flights(self):
        for (route, day), row in FORECAST.iterrows():
            info = ROUTES.loc[route]
            results = [calculate(row,info,float(p)) for p in [info.price_min,info.base_price,info.price_max]]
            self.assertGreaterEqual(results[0]['sales'],results[1]['sales'])
            self.assertGreaterEqual(results[1]['sales'],results[2]['sales'])
            for result in results:
                self.assertTrue(0 <= result['sales'] <= result['seats'])
                self.assertEqual(result['quantiles'][4]['demand'],result['demand'])
    def test_page_and_options(self):
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertEqual(len(self.client.get('/api/options').json()['routes']),5)

if __name__ == '__main__': unittest.main()
