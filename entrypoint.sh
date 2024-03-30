#!/bin/sh
# entrypoint.sh

# Start Flask application with dynamic PORT
flask run --host=0.0.0.0 --port=${PORT:-5000}
