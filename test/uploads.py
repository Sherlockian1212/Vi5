import requests

url = 'http://localhost:5000/upload'
files = {'file': open('/uploads/test01.jpg', 'rb')}

response = requests.post(url, files=files)

if response.status_code == 200:
    print('File uploaded successfully!')
    print(response.json())
else:
    print('Upload failed.')
    print(response.json())
