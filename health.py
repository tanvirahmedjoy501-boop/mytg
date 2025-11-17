from aiohttp import web
import os

async def health_check(request):
    return web.Response(text="OK")

if __name__ == "__main__":
    app = web.Application()
    app.router.add_get('/health', health_check)
    web.run_app(app, host='0.0.0.0', port=8080)
