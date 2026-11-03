"""Functional checks with isolated temporary SQLite, never a real user database."""
import time
import unittest
import uuid
from unittest.mock import patch
import test_api  # Initializes isolated database before importing app.
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import select, update
from app import app
from auth import AuthStore, COOKIE, RateLimiter, store


class TestAuth(unittest.TestCase):
    def setUp(self):
        store.limiter = RateLimiter()
        self.client = TestClient(app)
        self.data = {'username':'u_' + uuid.uuid4().hex[:16],
                     'display_name':'Проверка', 'password':'test password 123456'}

    def register(self):
        r = self.client.post('/api/auth/register', json=self.data)
        self.assertEqual(r.status_code, 201, r.text)
        self.client.headers['X-CSRF-Token'] = r.json()['csrf_token']
        return r

    def test_register_hash_cookie_and_session(self):
        r = self.register()
        self.assertEqual(r.json()['user']['display_name'], 'Проверка')
        self.assertNotIn('password', r.text)
        self.assertIn('HttpOnly', r.headers['set-cookie'])
        self.assertIn('SameSite=lax', r.headers['set-cookie'])
        self.assertEqual(r.headers['cache-control'], 'no-store')
        token = self.client.cookies[COOKIE]
        with store.engine.connect() as c:
            user = c.execute(select(store.users).where(
                store.users.c.username_key == self.data['username'])).mappings().one()
            session = c.execute(select(store.sessions).where(
                store.sessions.c.user_id == user['id'])).mappings().one()
        self.assertTrue(user['password_hash'].startswith('$argon2id$'))
        self.assertNotEqual(user['password_hash'], self.data['password'])
        self.assertNotEqual(session['token_hash'], token)
        self.assertTrue(store.hasher.verify(user['password_hash'], self.data['password']))
        self.assertEqual(self.client.get('/api/auth/me').status_code, 200)

    def test_case_insensitive_duplicate_username(self):
        self.register()
        r = self.client.post('/api/auth/register', json={**self.data, 'username':self.data['username'].upper()})
        self.assertEqual(r.status_code, 409)

    def test_invalid_registration(self):
        for change in [{'username':"x' OR 1=1 --"}, {'password':'short'}, {'display_name':'   '}]:
            self.assertEqual(self.client.post('/api/auth/register',json={**self.data, **change}).status_code,422)

    def test_wrong_password_and_unknown_account_same_message(self):
        self.register()
        a = self.client.post('/api/auth/login',json={**self.data,'password':'wrong password'})
        b = self.client.post('/api/auth/login',json={**self.data,'username':'unknown_user_123'})
        self.assertEqual(a.status_code,401)
        self.assertEqual(a.json(),b.json())

    def test_logout_revokes_cookie_and_login_restores_access(self):
        self.register()
        old_token = self.client.cookies[COOKIE]
        self.assertEqual(self.client.post('/api/auth/logout',json={}).status_code,204)
        self.assertEqual(self.client.get('/api/auth/me').status_code,401)
        stale = TestClient(app)
        stale.cookies.set(COOKIE,old_token)
        self.assertEqual(stale.get('/api/auth/me').status_code,401)
        login = self.client.post('/api/auth/login',json={**self.data, 'username':self.data['username'].upper()})
        self.assertEqual(login.status_code,200,login.text)
        self.assertNotEqual(old_token,self.client.cookies[COOKIE])

    def test_access_and_csrf_enforced_on_server(self):
        payload={'origin':'MOW','destination':'AER','date':'2026-06-12','price':10000}
        self.assertEqual(self.client.post('/api/predict',json=payload).status_code,401)
        self.register()
        self.assertEqual(self.client.post('/api/predict',json=payload).status_code,200)
        self.client.headers.pop('X-CSRF-Token')
        self.assertEqual(self.client.post('/api/predict',json=payload).status_code,403)
        self.assertEqual(self.client.post('/api/auth/logout',json={}).status_code,403)

    def test_cross_origin_requests_rejected(self):
        r=self.client.post('/api/auth/register',json=self.data,headers={'Origin':'https://foreign.example'})
        self.assertEqual(r.status_code,403)
        self.register()
        self.assertEqual(self.client.post('/api/auth/logout',json={},headers={'Origin':'https://foreign.example'}).status_code,403)
        self.assertEqual(self.client.post('/api/auth/login',json=self.data,headers={'Origin':'http://[broken'}).status_code,403)

    def test_expired_and_forged_sessions_rejected(self):
        self.client.cookies.set(COOKIE,'invented-token')
        self.assertEqual(self.client.get('/api/auth/me').status_code,401)
        self.client.cookies.clear()
        r=self.register()
        user_id=r.json()['user']['id']
        with store.engine.begin() as c:
            c.execute(update(store.sessions).where(store.sessions.c.user_id==user_id).values(expires_at=int(time.time())-1))
        self.assertEqual(self.client.get('/api/auth/me').status_code,401)

    def test_user_and_session_survive_new_store(self):
        self.register()
        second=AuthStore(str(store.engine.url))
        try:
            user=second.find_session(self.client.cookies[COOKIE])
            self.assertEqual(user['username'],self.data['username'])
        finally:
            second.engine.dispose()

    def test_login_rotates_existing_session(self):
        self.register()
        old=self.client.cookies[COOKIE]
        response=self.client.post('/api/auth/login',json=self.data)
        self.assertEqual(response.status_code,200)
        self.assertIsNone(store.find_session(old))
        self.assertIsNotNone(store.find_session(self.client.cookies[COOKIE]))

    def test_secure_cookie_configuration(self):
        with patch('auth.COOKIE_SECURE',True):
            client=TestClient(app,base_url='https://testserver')
            response=client.post('/api/auth/register',json=self.data)
            self.assertEqual(response.status_code,201)
            self.assertIn('Secure',response.headers['set-cookie'])
            self.assertEqual(client.get('/api/auth/me').status_code,200)

    def test_rate_limit_and_recovery(self):
        now=[0]
        limiter=RateLimiter(limit=2,window=60,clock=lambda:now[0])
        limiter.check('client');limiter.check('client')
        with self.assertRaises(HTTPException) as e:
            limiter.check('client')
        self.assertEqual(e.exception.status_code,429)
        now[0]=61
        limiter.check('client')

    def test_health_and_guest_page(self):
        self.assertEqual(self.client.get('/healthz').json()['status'],'ok')
        page=self.client.get('/')
        self.assertIn('authDialog',page.text)
        self.assertIn('Регистрация',page.text)
        self.assertEqual(self.client.get('/api/options').json()['demo'],False)


if __name__=='__main__':
    unittest.main()
