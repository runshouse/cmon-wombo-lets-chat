import aiohttp
import asyncio
import base64

encoded_string = ""
# # Open the file in binary mode
# with open('../latest.avif', "rb") as image_file:
#     # noinspection PyRedeclaration
#     encoded_string = base64.b64encode(image_file.read()).decode()

# Your API Gateway URL
url = 'https://fn7jwwghkbv23j32ed3bvzauci0evpln.lambda-url.us-east-2.on.aws/'

# Define asynchronous function
async def send_request(url, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data) as response:
            if await response.text() == 'Image uploaded to S3 successfully!':
                print('Succ3ss')

asyncio.run(send_request(url, data=encoded_string))
