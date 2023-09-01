import requests

url = 'http://localhost:5000/upload'
files = {'file': open('D:/STUDY/DHSP/NCKH-2023-With my idol/Vi6/uploads/test.jpg', 'rb')}

response = requests.post(url, files=files)

if response.status_code == 200:
    print('File uploaded successfully!')
    print(response.json())
else:
    print('Upload failed.')
    print(response.json())
