import asyncio
import nbapi

async def main():
    # First generate a service
    # service = nbapi.parse("https://gist.githubusercontent.com/samnm/df213c5ae4ddc6f5fdaba5e61ba1d877/raw/41d7b81416b61f9e9a8baf479330ce0448d1a727/simple.ipynb")
    # then edit service to configure the parameters, remove or reorder cells in the execution plan
    # then execute a service with input
    service = nbapi.Service.from_json(open("simple-example.service.json", 'r').read())
    await nbapi.exec(service, {
        "name": "'world'"
    })

asyncio.run(main())
