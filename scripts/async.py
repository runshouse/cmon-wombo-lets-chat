import aiohttp
import asyncio
async def main():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://6tdnw9xx0l.execute-api.us-east-2.amazonaws.com/default/animatediffcount') as resp:
            if resp.status == 200:
                print("Success")

if __name__ == "__main__":
    asyncio.run(main())

