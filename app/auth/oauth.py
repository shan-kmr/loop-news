from authlib.integrations.flask_client import OAuth
from flask import current_app

oauth = OAuth()

""" Simple set up of Google OAuth by setting up all the required parameters."""
def init_oauth(app):
    oauth.init_app(app)
    # Register Google provider
    oauth.register(
        name='google',
        client_id=app.config['GOOGLE_CLIENT_ID'],
        client_secret=app.config['GOOGLE_CLIENT_SECRET'],
        server_metadata_url=app.config['GOOGLE_DISCOVERY_URL'],
        client_kwargs={
            'scope': 'openid email profile'
        }
    )
    print("OAuth initialized and Google provider registered.") 