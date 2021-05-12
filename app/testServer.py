# ------------------------------------------------------
# Author : Naimish Mani B
# Date : 9th May 2021
# ------------------------------------------------------
# Tests the Flask API server
# ------------------------------------------------------

import requests


response = requests.post(
    'http://127.0.0.1:5000/api/processRequest?text="There are many movies we can watch"'
)
print(response.json())

# Check if the image looks right using the below URL
# https://codebeautify.org/base64-to-image-converter
